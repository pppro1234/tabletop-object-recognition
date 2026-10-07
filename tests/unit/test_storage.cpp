#include "factories.hpp"
#include "test_support.hpp"

#include <sqlite3.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using Database = std::unique_ptr<sqlite3, decltype(&sqlite3_close)>;
using Statement = std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)>;

Database openDatabase(const fs::path &path) {
    sqlite3 *raw = nullptr;
    const int result = sqlite3_open(path.string().c_str(), &raw);
    Database database(raw, sqlite3_close);
    vision_test::require(result == SQLITE_OK, "Failed to open the isolated test database.");
    return database;
}

void execute(sqlite3 *database, const std::string &sql) {
    char *error = nullptr;
    const int result = sqlite3_exec(database, sql.c_str(), nullptr, nullptr, &error);
    const std::string message = error == nullptr ? "" : error;
    sqlite3_free(error);
    vision_test::require(result == SQLITE_OK, "Fixture SQL failed: " + message);
}

std::vector<std::vector<std::string>> query(sqlite3 *database, const std::string &sql) {
    sqlite3_stmt *raw = nullptr;
    const int prepared = sqlite3_prepare_v2(database, sql.c_str(), -1, &raw, nullptr);
    Statement statement(raw, sqlite3_finalize);
    vision_test::require(prepared == SQLITE_OK, "Failed to query the test database.");

    std::vector<std::vector<std::string>> rows;
    int result = SQLITE_ROW;
    while ((result = sqlite3_step(statement.get())) == SQLITE_ROW) {
        std::vector<std::string> row;
        for (int column = 0; column < sqlite3_column_count(statement.get()); ++column) {
            const unsigned char *value = sqlite3_column_text(statement.get(), column);
            row.emplace_back(value == nullptr ? "" : reinterpret_cast<const char *>(value));
        }
        rows.push_back(std::move(row));
    }
    vision_test::require(result == SQLITE_DONE, "Failed while reading the test database.");
    return rows;
}

std::string readFile(const fs::path &path) {
    std::ifstream input(path, std::ios::binary);
    vision_test::require(input.is_open(), "Failed to open exported CSV: " + path.string());
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

double exportedAccuracy(const std::string &csv) {
    const std::string key = "accuracy_percent,";
    const std::size_t position = csv.rfind(key);
    vision_test::require(position != std::string::npos, "Confusion export should include accuracy.");
    return std::stod(csv.substr(position + key.size()));
}

void testLegacyMigration() {
    const vision::AppConfig config = vision_test::makeTestConfig("unit_storage_legacy");
    {
        Database database = openDatabase(config.storage.database_path);
        execute(database.get(),
                "CREATE TABLE object_samples (id INTEGER PRIMARY KEY AUTOINCREMENT, label TEXT NOT NULL, "
                "feature_vector TEXT NOT NULL, embedding_vector TEXT, image_path TEXT, created_at TEXT NOT NULL);"
                "CREATE TABLE runs (id INTEGER PRIMARY KEY AUTOINCREMENT, config_name TEXT NOT NULL, "
                "git_commit TEXT NOT NULL, dataset_name TEXT NOT NULL, started_at TEXT NOT NULL, "
                "avg_fps REAL, accuracy REAL);"
                "CREATE TABLE predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL, "
                "frame_id TEXT NOT NULL, true_label TEXT, pred_label TEXT NOT NULL, score REAL, latency_ms REAL, "
                "FOREIGN KEY(run_id) REFERENCES runs(id));"
                "INSERT INTO object_samples VALUES(7, 'camera', '0.5,2,-1,-3', '', 'original.jpg', '2026-03-09');"
                "INSERT INTO runs VALUES(17, 'original', 'old-commit', 'old-dataset', '2026-03-09', 30, 1);"
                "INSERT INTO predictions VALUES(41, 17, 'old-frame', 'camera', 'camera', 0.75, 1.25);");
    }

    // Opening an existing database twice must be safe and retain historical experiments.
    for (int opening = 0; opening < 2; ++opening) {
        const std::unique_ptr<vision::RunStorage> storage = vision::makeSqliteRunStorage(config);
        const std::vector<vision::TrainingSample> samples = storage->loadTrainingSamples();
        vision_test::require(samples.size() == 1 && samples.front().sample_id == "7" &&
                                 samples.front().label == "camera" && samples.front().image_path == "original.jpg",
                             "Migration should preserve the existing training sample.");
        vision_test::require(samples.front().feature.values == std::vector<double>({0.5, 2.0, -1.0, -3.0}),
                             "Migration should preserve historical feature values.");

        Database database = openDatabase(config.storage.database_path);
        const auto predictions = query(database.get(),
                                       "SELECT id, run_id, frame_id, true_label, pred_label, score, latency_ms, "
                                       "status, detection_count FROM predictions;");
        const std::vector<std::vector<std::string>> expected = {
            {"41", "17", "old-frame", "camera", "camera", "0.75", "1.25", "legacy", "0"}};
        vision_test::require(predictions == expected,
                             "Migration should preserve predictions and mark their missing provenance as legacy.");
        vision_test::require(query(database.get(), "SELECT config_name, git_commit, dataset_name FROM runs;") ==
                                 std::vector<std::vector<std::string>>({{"original", "old-commit", "old-dataset"}}),
                             "Migration should preserve run metadata.");
        vision_test::require(query(database.get(), "SELECT count(*) FROM pragma_table_info('predictions');") ==
                                 std::vector<std::vector<std::string>>({{"9"}}),
                             "Repeated initialization should not add duplicate columns.");

        const fs::path confusion = fs::path(config.evaluation.output_path).parent_path() / "legacy_confusion.csv";
        vision_test::require(storage->exportConfusionMatrix(17, confusion.string()),
                             "Historical predictions should remain exportable.");
        vision_test::requireNear(exportedAccuracy(readFile(confusion)), 100.0, 1e-6,
                                 "Legacy predictions should retain their historical correctness semantics.");
    }
}

void testEvaluationPersistenceAndCsv() {
    const vision::AppConfig config = vision_test::makeTestConfig("unit_storage_evaluation");
    std::unique_ptr<vision::RunStorage> storage = vision::makeSqliteRunStorage(config);
    vision::RunRecord run;
    run.config_name = "storage-test";
    run.dataset_name = "isolated-fixture";
    const std::int64_t run_id = storage->beginRun(run);
    vision_test::require(run_id > 0, "The evaluation run should be created.");

    vision::EvaluationRecord quoted;
    quoted.frame_id = "frames/set,\"A\".jpg";
    quoted.true_label = "part, \"left\"";
    quoted.predicted_label = quoted.true_label;
    quoted.score = 0.75f;
    quoted.latency_ms = 1.25;
    quoted.correct = true;
    quoted.status = "ok";
    quoted.detection_count = 2;

    vision::EvaluationRecord missing;
    missing.frame_id = "empty.jpg";
    missing.true_label = "Unknown";
    missing.predicted_label = "Unknown";
    missing.correct = false;
    missing.status = "no_detection";
    missing.detection_count = 0;

    vision::EvaluationRecord rejected = missing;
    rejected.frame_id = "unknown-object.jpg";
    rejected.correct = true;
    rejected.status = "ok";
    rejected.detection_count = 1;

    const std::vector<vision::EvaluationRecord> records = {quoted, missing, rejected};
    vision_test::require(storage->appendPredictions(run_id, records), "Evaluation predictions should be persisted.");
    storage.reset();
    storage = vision::makeSqliteRunStorage(config);

    Database database = openDatabase(config.storage.database_path);
    const auto persisted = query(database.get(),
                                 "SELECT frame_id, true_label, pred_label, status, detection_count "
                                 "FROM predictions ORDER BY id;");
    const std::vector<std::vector<std::string>> expected = {
        {quoted.frame_id, quoted.true_label, quoted.predicted_label, "ok", "2"},
        {"empty.jpg", "Unknown", "Unknown", "no_detection", "0"},
        {"unknown-object.jpg", "Unknown", "Unknown", "ok", "1"}};
    vision_test::require(persisted == expected, "Status, detection count and quoted text should survive reopening.");

    vision_test::require(storage->writeEvaluation(records), "Evaluation CSV should be written.");
    const std::string expected_csv =
        "frame_id,true_label,predicted_label,score,latency_ms,correct,status,detection_count\n"
        "\"frames/set,\"\"A\"\".jpg\",\"part, \"\"left\"\"\",\"part, \"\"left\"\"\",0.75,1.25,1,ok,2\n"
        "empty.jpg,Unknown,Unknown,0,0,0,no_detection,0\n"
        "unknown-object.jpg,Unknown,Unknown,0,0,1,ok,1\n";
    vision_test::require(readFile(config.evaluation.output_path) == expected_csv,
                         "Evaluation CSV must quote commas and double embedded quotes without shifting columns.");

    const fs::path confusion = fs::path(config.evaluation.output_path).parent_path() / "confusion.csv";
    vision_test::require(storage->exportConfusionMatrix(run_id, confusion.string()),
                         "Confusion matrix should be exported from persisted predictions.");
    const std::string confusion_csv = readFile(confusion);
    vision_test::require(confusion_csv.find("true/pred,Unknown,\"part, \"\"left\"\"\"\n") == 0 &&
                             confusion_csv.find("\"part, \"\"left\"\"\",0,1\n") != std::string::npos,
                         "Confusion CSV should escape labels in both its header and rows.");
    vision_test::requireNear(exportedAccuracy(confusion_csv), 200.0 / 3.0, 0.0001,
                             "A missing detection must count as an error even when the true label is Unknown.");
}

void testTrainingSampleWriteFailure() {
    const vision::AppConfig config = vision_test::makeTestConfig("unit_storage_sample_failure");
    const std::unique_ptr<vision::RunStorage> storage = vision::makeSqliteRunStorage(config);
    vision::TrainingSample sample;
    sample.label = "existing";
    sample.feature.values = {1.0, 2.0};
    sample.image_path = "existing.jpg";
    vision_test::require(storage->appendTrainingSample(sample), "The existing sample should be stored.");

    Database database = openDatabase(config.storage.database_path);
    execute(database.get(),
            "CREATE TRIGGER reject_sample BEFORE INSERT ON object_samples "
            "BEGIN SELECT RAISE(ABORT, 'injected sample write failure'); END;");
    sample.label = "new";
    sample.image_path = "new.jpg";
    bool failed = false;
    try {
        storage->appendTrainingSample(sample);
    } catch (const std::runtime_error &error) {
        failed = std::string(error.what()).find("injected sample write failure") != std::string::npos;
    }
    vision_test::require(failed, "Sample write errors should include the SQLite cause.");
    vision_test::require(query(database.get(), "SELECT label, image_path FROM object_samples ORDER BY id;") ==
                             std::vector<std::vector<std::string>>({{"existing", "existing.jpg"}}),
                         "A failed sample insert must leave existing rows intact.");

    execute(database.get(), "DROP TRIGGER reject_sample;");
    execute(database.get(), "PRAGMA journal_mode=DELETE;");
    execute(database.get(), "BEGIN;");
    vision_test::require(query(database.get(), "SELECT label FROM object_samples;").size() == 1,
                         "The reader should hold the existing sample snapshot.");
    failed = false;
    try {
        storage->appendTrainingSample(sample);
    } catch (const std::runtime_error &error) {
        failed = std::string(error.what()).find("locked") != std::string::npos;
    }
    vision_test::require(failed, "A blocked sample commit must report its SQLite cause.");
    execute(database.get(), "ROLLBACK;");
    vision_test::require(query(database.get(), "SELECT label, image_path FROM object_samples ORDER BY id;") ==
                             std::vector<std::vector<std::string>>({{"existing", "existing.jpg"}}),
                         "A blocked sample commit must leave the old row intact.");
    vision_test::require(storage->appendTrainingSample(sample), "A later sample write should succeed.");
    vision_test::require(storage->loadTrainingSamples().size() == 2,
                         "The sample store should remain usable after a write error.");
}

void testPredictionExecutionAndPrepareFailure() {
    const vision::AppConfig config = vision_test::makeTestConfig("unit_storage_prediction_failure");
    const std::unique_ptr<vision::RunStorage> storage = vision::makeSqliteRunStorage(config);
    vision::RunRecord run;
    run.config_name = "transaction-test";
    run.dataset_name = "isolated-fixture";
    const std::int64_t run_id = storage->beginRun(run);
    vision_test::require(run_id > 0, "The test run should be created.");

    vision::EvaluationRecord first;
    first.frame_id = "existing";
    first.predicted_label = "object";
    first.status = "ok";
    vision_test::require(storage->appendPredictions(run_id, {first}), "The existing prediction should be stored.");

    Database database = openDatabase(config.storage.database_path);
    execute(database.get(),
            "CREATE TRIGGER reject_prediction BEFORE INSERT ON predictions "
            "WHEN NEW.frame_id = 'blocked' BEGIN SELECT RAISE(ABORT, 'injected prediction failure'); END;");
    vision::EvaluationRecord allowed = first;
    allowed.frame_id = "uncommitted";
    vision::EvaluationRecord blocked = first;
    blocked.frame_id = "blocked";
    bool failed = false;
    try {
        storage->appendPredictions(run_id, {allowed, blocked});
    } catch (const std::runtime_error &error) {
        failed = std::string(error.what()).find("injected prediction failure") != std::string::npos;
    }
    vision_test::require(failed, "Prediction execution errors should include the SQLite cause.");
    vision_test::require(query(database.get(), "SELECT frame_id FROM predictions ORDER BY id;") ==
                             std::vector<std::vector<std::string>>({{"existing"}}),
                         "A failed prediction batch must roll back earlier rows in that batch.");

    execute(database.get(), "DROP TRIGGER reject_prediction;");
    execute(database.get(), "ALTER TABLE predictions RENAME TO hidden_predictions;");
    failed = false;
    try {
        storage->appendPredictions(run_id, {allowed});
    } catch (const std::runtime_error &error) {
        failed = std::string(error.what()).find("no such table") != std::string::npos;
    }
    vision_test::require(failed, "A prepare error should report its SQLite cause.");
    execute(database.get(), "ALTER TABLE hidden_predictions RENAME TO predictions;");
    vision_test::require(storage->appendPredictions(run_id, {allowed}),
                         "Prediction writes should recover after execution and prepare errors.");
    vision_test::require(query(database.get(), "SELECT frame_id FROM predictions ORDER BY id;") ==
                             std::vector<std::vector<std::string>>({{"existing"}, {"uncommitted"}}),
                         "Failure recovery should preserve old rows and persist only the successful retry.");
}

void testPredictionCommitFailure() {
    const vision::AppConfig config = vision_test::makeTestConfig("unit_storage_commit_failure");
    const std::unique_ptr<vision::RunStorage> storage = vision::makeSqliteRunStorage(config);
    vision::RunRecord run;
    run.config_name = "commit-test";
    run.dataset_name = "isolated-fixture";
    const std::int64_t run_id = storage->beginRun(run);
    vision_test::require(run_id > 0, "The test run should be created.");

    vision::EvaluationRecord record;
    record.frame_id = "existing";
    record.predicted_label = "object";
    record.status = "ok";
    vision_test::require(storage->appendPredictions(run_id, {record}), "The existing prediction should be stored.");

    Database reader = openDatabase(config.storage.database_path);
    execute(reader.get(), "PRAGMA journal_mode=DELETE;");
    execute(reader.get(), "BEGIN;");
    vision_test::require(query(reader.get(), "SELECT frame_id FROM predictions;") ==
                             std::vector<std::vector<std::string>>({{"existing"}}),
                         "The reader should hold a snapshot of existing predictions.");

    record.frame_id = "blocked-commit";
    bool failed = false;
    try {
        storage->appendPredictions(run_id, {record});
    } catch (const std::runtime_error &error) {
        const std::string message = error.what();
        failed = message.find("commit prediction transaction") != std::string::npos &&
                 message.find("locked") != std::string::npos;
    }
    vision_test::require(failed, "A locked COMMIT must report failure with the SQLite cause.");
    execute(reader.get(), "ROLLBACK;");
    vision_test::require(query(reader.get(), "SELECT frame_id FROM predictions ORDER BY id;") ==
                             std::vector<std::vector<std::string>>({{"existing"}}),
                         "A failed COMMIT must not persist its new prediction.");

    record.frame_id = "later-success";
    vision_test::require(storage->appendPredictions(run_id, {record}),
                         "The same storage connection should support a later successful transaction.");
    vision_test::require(query(reader.get(), "SELECT frame_id FROM predictions ORDER BY id;") ==
                             std::vector<std::vector<std::string>>({{"existing"}, {"later-success"}}),
                         "A later write should preserve old rows after a failed COMMIT.");
}

}  // namespace

int main() {
    testLegacyMigration();
    testEvaluationPersistenceAndCsv();
    testTrainingSampleWriteFailure();
    testPredictionExecutionAndPrepareFailure();
    testPredictionCommitFailure();
    return 0;
}
