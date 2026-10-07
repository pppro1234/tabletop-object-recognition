#include "factories.hpp"

#include <sqlite3.h>

#include <algorithm>
#include <cctype>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vision {
namespace {

namespace fs = std::filesystem;

std::string trimCopy(const std::string &value) {
    std::size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) {
        ++begin;
    }

    std::size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return value.substr(begin, end - begin);
}

std::vector<double> parseVector(const std::string &encoded) {
    std::vector<double> values;
    std::stringstream stream(encoded);
    std::string item;
    while (std::getline(stream, item, ',')) {
        const std::string token = trimCopy(item);
        if (!token.empty()) {
            values.push_back(std::stod(token));
        }
    }
    return values;
}

std::string serializeVector(const std::vector<double> &values) {
    std::ostringstream stream;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            stream << ",";
        }
        stream << std::setprecision(12) << values[i];
    }
    return stream.str();
}

std::string isoTimestampUtc() {
    const std::time_t raw_time = std::time(nullptr);
    std::tm time_info{};
#if defined(_WIN32)
    gmtime_s(&time_info, &raw_time);
#else
    gmtime_r(&raw_time, &time_info);
#endif

    std::ostringstream stream;
    stream << std::put_time(&time_info, "%Y-%m-%dT%H:%M:%SZ");
    return stream.str();
}

std::string textColumn(sqlite3_stmt *statement, int column) {
    const unsigned char *text = sqlite3_column_text(statement, column);
    return (text == nullptr) ? std::string() : reinterpret_cast<const char *>(text);
}

std::string csvField(const std::string &value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) {
        return value;
    }
    std::string escaped = "\"";
    for (char ch : value) {
        if (ch == '"') {
            escaped.push_back('"');
        }
        escaped.push_back(ch);
    }
    return escaped + '"';
}

class SqliteRunStorage final : public RunStorage {
public:
    explicit SqliteRunStorage(const AppConfig &config)
        : database_path_(config.storage.database_path), evaluation_path_(config.evaluation.output_path) {
        openDatabase();
        try {
            ensureSchema();
        } catch (...) {
            sqlite3_close(db_);
            db_ = nullptr;
            throw;
        }
    }

    ~SqliteRunStorage() override {
        if (db_ != nullptr) {
            sqlite3_close(db_);
        }
    }

    std::vector<TrainingSample> loadTrainingSamples() const override {
        std::vector<TrainingSample> samples;
        sqlite3_stmt *statement = prepare(
            "SELECT id, label, feature_vector, embedding_vector, image_path, created_at FROM object_samples ORDER BY id;");

        while (sqlite3_step(statement) == SQLITE_ROW) {
            TrainingSample sample;
            sample.sample_id = std::to_string(sqlite3_column_int64(statement, 0));
            sample.label = textColumn(statement, 1);
            sample.feature.extractor = "shape4d";
            sample.feature.values = parseVector(textColumn(statement, 2));
            sample.embedding = parseVector(textColumn(statement, 3));
            sample.image_path = textColumn(statement, 4);
            sample.created_at = textColumn(statement, 5);
            if (!sample.label.empty() && !sample.feature.values.empty()) {
                samples.push_back(std::move(sample));
            }
        }

        sqlite3_finalize(statement);
        return samples;
    }

    bool appendTrainingSample(const TrainingSample &sample) override {
        const std::string created_at = sample.created_at.empty() ? isoTimestampUtc() : sample.created_at;
        const std::string feature_text = serializeVector(sample.feature.values);
        const std::string embedding_text = serializeVector(sample.embedding);
        std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> statement(
            prepare("INSERT INTO object_samples(label, feature_vector, embedding_vector, image_path, created_at) "
                    "VALUES(?, ?, ?, ?, ?);"),
            sqlite3_finalize);
        if (sqlite3_bind_text(statement.get(), 1, sample.label.c_str(), -1, SQLITE_TRANSIENT) != SQLITE_OK ||
            sqlite3_bind_text(statement.get(), 2, feature_text.c_str(), -1, SQLITE_TRANSIENT) != SQLITE_OK ||
            sqlite3_bind_text(statement.get(), 3, embedding_text.c_str(), -1, SQLITE_TRANSIENT) != SQLITE_OK ||
            sqlite3_bind_text(statement.get(), 4, sample.image_path.c_str(), -1, SQLITE_TRANSIENT) != SQLITE_OK ||
            sqlite3_bind_text(statement.get(), 5, created_at.c_str(), -1, SQLITE_TRANSIENT) != SQLITE_OK) {
            throw std::runtime_error("Failed to bind training sample: " + std::string(sqlite3_errmsg(db_)));
        }
        if (sqlite3_step(statement.get()) != SQLITE_DONE) {
            throw std::runtime_error("Failed to insert training sample: " + std::string(sqlite3_errmsg(db_)));
        }
        const int finalize_result = sqlite3_finalize(statement.release());
        if (finalize_result != SQLITE_OK) {
            std::string detail = sqlite3_errmsg(db_);
            if (!sqlite3_get_autocommit(db_)) {
                std::string rollback_error;
                if (!execSql("ROLLBACK;", &rollback_error)) {
                    detail += "; rollback failed: " + rollback_error;
                }
            }
            throw std::runtime_error("Failed to finish training sample insert: " + detail);
        }
        return true;
    }

    std::int64_t beginRun(const RunRecord &run) override {
        sqlite3_stmt *statement = prepare(
            "INSERT INTO runs(config_name, git_commit, dataset_name, started_at, avg_fps, accuracy) VALUES(?, ?, ?, ?, NULL, NULL);");

        const std::string started_at = run.started_at.empty() ? isoTimestampUtc() : run.started_at;
        sqlite3_bind_text(statement, 1, run.config_name.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(statement, 2, run.git_commit.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(statement, 3, run.dataset_name.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(statement, 4, started_at.c_str(), -1, SQLITE_TRANSIENT);

        if (sqlite3_step(statement) != SQLITE_DONE) {
            sqlite3_finalize(statement);
            return 0;
        }

        sqlite3_finalize(statement);
        return sqlite3_last_insert_rowid(db_);
    }

    bool appendPredictions(std::int64_t run_id, const std::vector<EvaluationRecord> &records) override {
        std::string sql_error;
        if (!execSql("BEGIN TRANSACTION;", &sql_error)) {
            throw std::runtime_error("Failed to begin prediction transaction: " + sql_error);
        }

        try {
            std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> statement(
                prepare("INSERT INTO predictions(run_id, frame_id, true_label, pred_label, score, latency_ms, status, detection_count) "
                        "VALUES(?, ?, ?, ?, ?, ?, ?, ?);"),
                sqlite3_finalize);

            for (const EvaluationRecord &record : records) {
                if (sqlite3_bind_int64(statement.get(), 1, run_id) != SQLITE_OK ||
                    sqlite3_bind_text(statement.get(), 2, record.frame_id.c_str(), -1, SQLITE_TRANSIENT) != SQLITE_OK ||
                    sqlite3_bind_text(statement.get(), 3, record.true_label.c_str(), -1, SQLITE_TRANSIENT) != SQLITE_OK ||
                    sqlite3_bind_text(statement.get(), 4, record.predicted_label.c_str(), -1, SQLITE_TRANSIENT) != SQLITE_OK ||
                    sqlite3_bind_double(statement.get(), 5, record.score) != SQLITE_OK ||
                    sqlite3_bind_double(statement.get(), 6, record.latency_ms) != SQLITE_OK ||
                    sqlite3_bind_text(statement.get(), 7, record.status.c_str(), -1, SQLITE_TRANSIENT) != SQLITE_OK ||
                    sqlite3_bind_int(statement.get(), 8, record.detection_count) != SQLITE_OK) {
                    throw std::runtime_error("Failed to bind prediction row: " + std::string(sqlite3_errmsg(db_)));
                }
                if (sqlite3_step(statement.get()) != SQLITE_DONE) {
                    throw std::runtime_error("Failed to insert prediction row: " + std::string(sqlite3_errmsg(db_)));
                }
                if (sqlite3_reset(statement.get()) != SQLITE_OK ||
                    sqlite3_clear_bindings(statement.get()) != SQLITE_OK) {
                    throw std::runtime_error("Failed to reset prediction statement: " + std::string(sqlite3_errmsg(db_)));
                }
            }

            statement.reset();
            if (!execSql("COMMIT;", &sql_error)) {
                throw std::runtime_error("Failed to commit prediction transaction: " + sql_error);
            }
        } catch (const std::exception &error) {
            const std::string original_error = error.what();
            if (!sqlite3_get_autocommit(db_)) {
                if (!execSql("ROLLBACK;", &sql_error)) {
                    throw std::runtime_error(original_error + "; failed to roll back prediction transaction: " + sql_error);
                }
            }
            throw;
        } catch (...) {
            if (!sqlite3_get_autocommit(db_)) {
                if (!execSql("ROLLBACK;", &sql_error)) {
                    throw std::runtime_error("Failed to roll back prediction transaction: " + sql_error);
                }
            }
            throw;
        }
        return true;
    }

    bool finishRun(std::int64_t run_id, double avg_fps, double accuracy) override {
        sqlite3_stmt *statement = prepare("UPDATE runs SET avg_fps = ?, accuracy = ? WHERE id = ?;");
        sqlite3_bind_double(statement, 1, avg_fps);
        sqlite3_bind_double(statement, 2, accuracy);
        sqlite3_bind_int64(statement, 3, run_id);

        const bool ok = sqlite3_step(statement) == SQLITE_DONE;
        sqlite3_finalize(statement);
        return ok;
    }

    bool exportConfusionMatrix(std::int64_t run_id, const std::string &csv_path) const override {
        sqlite3_stmt *statement =
            prepare("SELECT true_label, pred_label, status FROM predictions WHERE run_id = ? AND true_label <> '';");
        sqlite3_bind_int64(statement, 1, run_id);

        std::vector<std::pair<std::string, std::string>> rows;
        std::set<std::string> label_set;
        int correct = 0;
        while (sqlite3_step(statement) == SQLITE_ROW) {
            const std::string true_label = textColumn(statement, 0);
            const std::string pred_label = textColumn(statement, 1);
            const std::string status = textColumn(statement, 2);
            if (true_label == pred_label && (status == "ok" || status == "legacy")) {
                ++correct;
            }
            rows.emplace_back(true_label, pred_label);
            label_set.insert(true_label);
            label_set.insert(pred_label);
        }
        sqlite3_finalize(statement);

        std::vector<std::string> labels(label_set.begin(), label_set.end());
        std::vector<std::vector<int>> matrix(labels.size(), std::vector<int>(labels.size(), 0));

        auto indexOf = [&labels](const std::string &label) {
            return static_cast<int>(std::distance(labels.begin(), std::find(labels.begin(), labels.end(), label)));
        };

        int total = 0;
        for (const auto &[true_label, pred_label] : rows) {
            const int row = indexOf(true_label);
            const int column = indexOf(pred_label);
            if (row < 0 || column < 0 || row >= static_cast<int>(labels.size()) || column >= static_cast<int>(labels.size())) {
                continue;
            }
            ++matrix[static_cast<std::size_t>(row)][static_cast<std::size_t>(column)];
            ++total;
        }

        std::error_code ec;
        fs::create_directories(fs::path(csv_path).parent_path(), ec);
        std::ofstream output(csv_path);
        if (!output.is_open()) {
            return false;
        }

        output << "true/pred";
        for (const std::string &label : labels) {
            output << "," << csvField(label);
        }
        output << "\n";

        for (std::size_t row = 0; row < labels.size(); ++row) {
            output << csvField(labels[row]);
            for (std::size_t column = 0; column < labels.size(); ++column) {
                output << "," << matrix[row][column];
            }
            output << "\n";
        }

        const double accuracy = (total > 0) ? (100.0 * static_cast<double>(correct) / total) : 0.0;
        output << "accuracy_percent," << accuracy << "\n";
        return true;
    }

    bool writeEvaluation(const std::vector<EvaluationRecord> &records) const override {
        std::error_code ec;
        fs::create_directories(fs::path(evaluation_path_).parent_path(), ec);
        std::ofstream output(evaluation_path_);
        if (!output.is_open()) {
            return false;
        }

        output << "frame_id,true_label,predicted_label,score,latency_ms,correct,status,detection_count\n";
        output << std::setprecision(12);
        for (const EvaluationRecord &record : records) {
            output << csvField(record.frame_id) << "," << csvField(record.true_label) << ","
                   << csvField(record.predicted_label) << "," << record.score << "," << record.latency_ms << ","
                   << (record.correct ? 1 : 0) << "," << csvField(record.status) << "," << record.detection_count << "\n";
        }
        return true;
    }

private:
    void openDatabase() {
        std::error_code ec;
        fs::create_directories(fs::path(database_path_).parent_path(), ec);
        if (sqlite3_open(database_path_.c_str(), &db_) != SQLITE_OK) {
            const std::string message = (db_ == nullptr) ? "sqlite3_open failed" : sqlite3_errmsg(db_);
            if (db_ != nullptr) {
                sqlite3_close(db_);
                db_ = nullptr;
            }
            throw std::runtime_error("Failed to open database: " + database_path_ + " (" + message + ")");
        }
    }

    void ensureSchema() {
        requireSql(
            "CREATE TABLE IF NOT EXISTS object_samples ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "label TEXT NOT NULL,"
            "feature_vector TEXT NOT NULL,"
            "embedding_vector TEXT,"
            "image_path TEXT,"
            "created_at TEXT NOT NULL"
            ");");
        requireSql(
            "CREATE TABLE IF NOT EXISTS runs ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "config_name TEXT NOT NULL,"
            "git_commit TEXT NOT NULL,"
            "dataset_name TEXT NOT NULL,"
            "started_at TEXT NOT NULL,"
            "avg_fps REAL,"
            "accuracy REAL"
            ");");
        requireSql(
            "CREATE TABLE IF NOT EXISTS predictions ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "run_id INTEGER NOT NULL,"
            "frame_id TEXT NOT NULL,"
            "true_label TEXT,"
            "pred_label TEXT NOT NULL,"
            "score REAL,"
            "latency_ms REAL,"
            "FOREIGN KEY(run_id) REFERENCES runs(id)"
            ");");
        // Preserve existing experiments while distinguishing rows written before status tracking.
        if (!hasPredictionColumn("status")) {
            requireSql("ALTER TABLE predictions ADD COLUMN status TEXT NOT NULL DEFAULT 'legacy';");
        }
        if (!hasPredictionColumn("detection_count")) {
            requireSql("ALTER TABLE predictions ADD COLUMN detection_count INTEGER NOT NULL DEFAULT 0;");
        }
        requireSql("CREATE INDEX IF NOT EXISTS idx_predictions_run_id ON predictions(run_id);");
    }

    bool hasPredictionColumn(const std::string &name) const {
        sqlite3_stmt *statement = prepare("PRAGMA table_info(predictions);");
        bool found = false;
        while (sqlite3_step(statement) == SQLITE_ROW) {
            found = found || textColumn(statement, 1) == name;
        }
        sqlite3_finalize(statement);
        return found;
    }

    void requireSql(const std::string &sql) const {
        std::string sql_error;
        if (!execSql(sql, &sql_error)) {
            throw std::runtime_error("Failed to initialize database schema: " + database_path_ + " (" + sql_error + ")");
        }
    }

    bool execSql(const std::string &sql, std::string *error = nullptr) const {
        char *error_message = nullptr;
        const int rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &error_message);
        if (rc != SQLITE_OK) {
            if (error != nullptr) {
                *error = error_message == nullptr ? sqlite3_errmsg(db_) : error_message;
            }
            sqlite3_free(error_message);
            return false;
        }
        sqlite3_free(error_message);
        return true;
    }

    sqlite3_stmt *prepare(const std::string &sql) const {
        sqlite3_stmt *statement = nullptr;
        if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &statement, nullptr) != SQLITE_OK) {
            const std::string error = sqlite3_errmsg(db_);
            sqlite3_finalize(statement);
            throw std::runtime_error("Failed to prepare SQL statement for database: " + database_path_ + " (" + error + ")");
        }
        return statement;
    }

    std::string database_path_;
    std::string evaluation_path_;
    mutable sqlite3 *db_ = nullptr;
};

}  // namespace

std::unique_ptr<RunStorage> makeSqliteRunStorage(const AppConfig &config) {
    return std::make_unique<SqliteRunStorage>(config);
}

}  // namespace vision
