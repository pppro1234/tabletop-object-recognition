#include "test_support.hpp"
#include "vision/pipeline.hpp"

#include <opencv2/imgcodecs.hpp>

#include <sqlite3.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;

std::vector<fs::path> savedImages(const fs::path &root) {
    std::vector<fs::path> paths;
    if (fs::exists(root)) {
        for (const auto &entry : fs::recursive_directory_iterator(root)) {
            if (entry.is_regular_file() && entry.path().extension() == ".png") {
                paths.push_back(entry.path());
            }
        }
    }
    return paths;
}

void requirePixel(const fs::path &path, unsigned char value) {
    const cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
    vision_test::require(!image.empty(), "Saved training image is unreadable: " + path.string());
    vision_test::require(image.at<cv::Vec3b>(0, 0)[0] == value, "An older training image was overwritten.");
}

vision::FramePacket frameFor(const std::string &source, unsigned char value) {
    return vision_test::makeFrame(source, 7, cv::Mat(16, 16, CV_8UC3, cv::Scalar(value, value, value)));
}

void execSql(sqlite3 *db, const char *sql) {
    char *message = nullptr;
    const int rc = sqlite3_exec(db, sql, nullptr, nullptr, &message);
    const std::string detail = message == nullptr ? sqlite3_errmsg(db) : message;
    sqlite3_free(message);
    vision_test::require(rc == SQLITE_OK, "SQLite test setup failed: " + detail);
}

}  // namespace

int main() {
    vision::AppConfig config = vision_test::makeTestConfig("integration_training_persistence");
    std::unique_ptr<vision::RunStorage> storage = vision::makeRunStorage(config);
    std::unique_ptr<vision::Pipeline> pipeline = vision::makeDefaultPipeline(config, *storage);

    vision::PipelineResult result;
    result.candidates.resize(1);
    result.candidates[0].feature.values = {1.0, 2.0, 3.0, 4.0};
    result.candidates[0].region.region_id = 1;
    result.candidates[0].detection.bounding_box = cv::Rect(0, 0, 16, 16);

    const fs::path first_source = fs::path(config.storage.sample_image_dir).parent_path() / "first" / "same.ppm";
    const fs::path second_source = fs::path(config.storage.sample_image_dir).parent_path() / "second" / "same.ppm";
    std::string error;
    for (const auto &[source, color] : std::vector<std::pair<fs::path, unsigned char>>{
             {first_source, 10}, {first_source, 40}, {second_source, 70}}) {
        const bool saved = pipeline->storeTrainingSample(frameFor(source.string(), color), result, 0, "object", &error);
        vision_test::require(saved, "Normal or repeated save failed: " + error);
    }

    const std::vector<vision::TrainingSample> before = storage->loadTrainingSamples();
    vision_test::require(before.size() == 3 && pipeline->classifier().sampleCount() == 3,
                         "Each repeated or same-name input needs its own database record and in-memory sample.");
    vision_test::require(before[0].image_path != before[1].image_path &&
                             before[0].image_path != before[2].image_path &&
                             before[1].image_path != before[2].image_path,
                         "Sample image paths must be unique.");
    for (std::size_t index = 0; index < before.size(); ++index) {
        requirePixel(before[index].image_path, static_cast<unsigned char>(10 + 30 * index));
    }
    vision_test::require(savedImages(config.storage.sample_image_dir).size() == 3,
                         "Normal saves must create three images.");

    // A regular file in place of the sample root gives a deterministic image-save failure.
    const fs::path blocked_root = fs::path(config.storage.sample_image_dir).parent_path() / "blocked";
    std::ofstream(blocked_root) << "not a directory";
    vision::AppConfig blocked_config = config;
    blocked_config.storage.sample_image_dir = blocked_root.string();
    std::unique_ptr<vision::Pipeline> blocked_pipeline = vision::makeDefaultPipeline(blocked_config, *storage);
    vision_test::require(!blocked_pipeline->storeTrainingSample(frameFor(first_source.string(), 90), result, 0,
                                                               "object", &error),
                         "Image-save failure must not report success.");
    vision_test::require(error.find("sample directory") != std::string::npos,
                         "Image-save failure should explain the failing path.");
    vision_test::require(storage->loadTrainingSamples().size() == 3 && blocked_pipeline->classifier().sampleCount() == 3,
                         "Image-save failure must leave the database and classifier unchanged.");

    // A SQLite trigger makes the next insert fail without relying on timing, permissions, or disk state.
    sqlite3 *db = nullptr;
    vision_test::require(sqlite3_open(config.storage.database_path.c_str(), &db) == SQLITE_OK,
                         "Could not open temporary SQLite database for failure test.");
    execSql(db, "CREATE TRIGGER reject_sample BEFORE INSERT ON object_samples "
                "BEGIN SELECT RAISE(FAIL, 'injected training insert failure'); END;");
    vision_test::require(!pipeline->storeTrainingSample(frameFor(first_source.string(), 100), result, 0,
                                                        "object", &error),
                         "Database-write failure must not report success.");
    vision_test::require(error.find("injected training insert failure") != std::string::npos,
                         "Database-write failure should preserve the SQLite error.");
    vision_test::require(storage->loadTrainingSamples().size() == 3 && pipeline->classifier().sampleCount() == 3,
                         "Database-write failure must leave committed and in-memory training state unchanged.");
    vision_test::require(savedImages(config.storage.sample_image_dir).size() == 3,
                         "Database-write failure must remove only its newly created image.");
    for (std::size_t index = 0; index < before.size(); ++index) {
        requirePixel(before[index].image_path, static_cast<unsigned char>(10 + 30 * index));
    }

    execSql(db, "DROP TRIGGER reject_sample;");
    sqlite3_close(db);
    const bool recovered = pipeline->storeTrainingSample(frameFor(second_source.string(), 110), result, 0,
                                                         "object", &error);
    vision_test::require(recovered, "A normal save should succeed after a database failure: " + error);
    vision_test::require(storage->loadTrainingSamples().size() == 4 && pipeline->classifier().sampleCount() == 4,
                         "Recovery save must add one database and in-memory sample.");
    vision_test::require(savedImages(config.storage.sample_image_dir).size() == 4,
                         "Recovery save must add one new image.");
    return 0;
}
