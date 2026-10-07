#include "factories.hpp"

#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <vector>

namespace vision {
namespace {

double distanceSquared(const cv::Point2f &a, const cv::Point2f &b) {
    const double dx = static_cast<double>(a.x - b.x);
    const double dy = static_cast<double>(a.y - b.y);
    return dx * dx + dy * dy;
}

class CentroidTracker final : public Tracker {
public:
    explicit CentroidTracker(TrackingConfig config) : config_(config) {}

    std::vector<Detection> assign(const std::vector<Detection> &detections) override {
        std::vector<Detection> tracked = detections;
        std::map<int, cv::Point2f> next_tracks;
        std::set<int> used_tracks;
        const double max_distance2 = static_cast<double>(config_.max_match_distance) * config_.max_match_distance;

        for (Detection &detection : tracked) {
            int best_track_id = -1;
            double best_distance2 = std::numeric_limits<double>::infinity();
            for (const auto &[track_id, centroid] : previous_tracks_) {
                if (used_tracks.count(track_id) > 0) {
                    continue;
                }
                const double current_distance2 = distanceSquared(detection.centroid, centroid);
                if (current_distance2 < best_distance2) {
                    best_distance2 = current_distance2;
                    best_track_id = track_id;
                }
            }

            if (best_track_id >= 0 && best_distance2 <= max_distance2) {
                detection.track_id = best_track_id;
                used_tracks.insert(best_track_id);
            } else {
                detection.track_id = next_track_id_++;
            }
            next_tracks[detection.track_id] = detection.centroid;
        }

        previous_tracks_ = std::move(next_tracks);
        return tracked;
    }

    void reset() override {
        previous_tracks_.clear();
        next_track_id_ = 1;
    }

private:
    TrackingConfig config_;
    std::map<int, cv::Point2f> previous_tracks_;
    int next_track_id_ = 1;
};

}  // namespace

std::unique_ptr<Tracker> makeCentroidTracker(const TrackingConfig &config) {
    return std::make_unique<CentroidTracker>(config);
}

}  // namespace vision
