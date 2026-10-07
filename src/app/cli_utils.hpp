#pragma once

#include "vision/types.hpp"

#include <cctype>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace vision::app {

struct ParsedArgs {
    std::map<std::string, std::string> options;
    std::vector<std::string> positionals;
};

inline ParsedArgs parseArgs(int argc, char **argv) {
    ParsedArgs parsed;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.rfind("--", 0) != 0) {
            parsed.positionals.push_back(arg);
            continue;
        }

        const std::size_t equals_pos = arg.find('=');
        if (equals_pos != std::string::npos) {
            parsed.options[arg.substr(2, equals_pos - 2)] = arg.substr(equals_pos + 1);
            continue;
        }

        const std::string key = arg.substr(2);
        if (i + 1 < argc) {
            const std::string next = argv[i + 1];
            if (next.rfind("--", 0) != 0) {
                parsed.options[key] = next;
                ++i;
                continue;
            }
        }

        parsed.options[key] = "true";
    }
    return parsed;
}

inline bool hasOption(const ParsedArgs &parsed, const std::string &key) {
    return parsed.options.find(key) != parsed.options.end();
}

inline std::string optionOr(const ParsedArgs &parsed, const std::string &key, const std::string &fallback) {
    const auto it = parsed.options.find(key);
    return (it == parsed.options.end()) ? fallback : it->second;
}

inline std::string requireOption(const ParsedArgs &parsed, const std::string &key) {
    const auto it = parsed.options.find(key);
    if (it == parsed.options.end() || it->second.empty() || it->second == "true") {
        throw std::runtime_error("Missing required option --" + key);
    }
    return it->second;
}

inline void applyStorageOverrides(AppConfig &config, const ParsedArgs &parsed) {
    if (hasOption(parsed, "database")) {
        config.storage.database_path = requireOption(parsed, "database");
    }
    if (hasOption(parsed, "artifacts")) {
        config.observability.artifact_root = requireOption(parsed, "artifacts");
    }
    if (hasOption(parsed, "sample-dir")) {
        config.storage.sample_image_dir = requireOption(parsed, "sample-dir");
    }
}

inline bool isIntegerString(const std::string &value) {
    if (value.empty()) {
        return false;
    }
    for (char ch : value) {
        if (!std::isdigit(static_cast<unsigned char>(ch))) {
            return false;
        }
    }
    return true;
}

inline void applySourceOverride(AppConfig &config, const std::string &source) {
    if (source.empty()) {
        return;
    }

    if (source == "webcam") {
        config.input.mode = InputMode::kWebcam;
        config.input.webcam_index = 0;
        return;
    }

    if (isIntegerString(source)) {
        config.input.mode = InputMode::kWebcam;
        config.input.webcam_index = std::stoi(source);
        return;
    }

    config.input.mode = InputMode::kBatchImages;
    config.input.image_dir = source;
}

}  // namespace vision::app
