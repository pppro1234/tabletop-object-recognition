#include "vision/interfaces.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace vision {
namespace {

namespace fs = std::filesystem;

double nowMs() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

std::string lowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

bool matchesImageFilter(const fs::path &path, const std::string &glob) {
    const std::string ext = lowerCopy(path.extension().string());
    if (glob.size() > 2 && glob[0] == '*' && glob[1] == '.') {
        return ext == lowerCopy(glob.substr(1));
    }
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

class WebcamFrameSource final : public FrameSource {
public:
    bool open(const InputConfig &config) override {
        close();
        source_id_ = "webcam:" + std::to_string(config.webcam_index);
        frame_index_ = 0;
        return capture_.open(config.webcam_index);
    }

    bool next(FramePacket &frame) override {
        cv::Mat image;
        capture_ >> image;
        if (image.empty()) {
            return false;
        }

        frame.source_id = source_id_;
        frame.frame_index = frame_index_++;
        frame.timestamp_ms = nowMs();
        frame.frame = image;
        return true;
    }

    void close() override {
        if (capture_.isOpened()) {
            capture_.release();
        }
    }

private:
    cv::VideoCapture capture_;
    std::string source_id_;
    std::int64_t frame_index_ = 0;
};

class BatchImageFrameSource final : public FrameSource {
public:
    bool open(const InputConfig &config) override {
        images_.clear();
        cursor_ = 0;

        const fs::path image_path(config.image_dir);
        std::error_code ec;
        if (fs::is_regular_file(image_path, ec)) {
            images_.push_back(image_path);
        } else if (fs::is_directory(image_path, ec)) {
            for (const auto &entry : fs::directory_iterator(image_path, ec)) {
                if (!entry.is_regular_file()) {
                    continue;
                }
                if (matchesImageFilter(entry.path(), config.image_glob)) {
                    images_.push_back(entry.path());
                }
            }
        }

        std::sort(images_.begin(), images_.end());
        return !images_.empty();
    }

    bool next(FramePacket &frame) override {
        while (cursor_ < images_.size()) {
            const fs::path path = images_[cursor_++];
            cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
            if (image.empty()) {
                continue;
            }

            frame.source_id = path.string();
            frame.frame_index = static_cast<std::int64_t>(cursor_ - 1);
            frame.timestamp_ms = nowMs();
            frame.frame = std::move(image);
            return true;
        }
        return false;
    }

    void close() override {
        images_.clear();
        cursor_ = 0;
    }

private:
    std::vector<fs::path> images_;
    std::size_t cursor_ = 0;
};

}  // namespace

std::unique_ptr<FrameSource> makeWebcamFrameSource() {
    return std::make_unique<WebcamFrameSource>();
}

std::unique_ptr<FrameSource> makeBatchImageFrameSource() {
    return std::make_unique<BatchImageFrameSource>();
}

}  // namespace vision
