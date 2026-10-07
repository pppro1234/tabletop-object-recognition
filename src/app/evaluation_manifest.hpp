#pragma once

#include <cctype>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vision::app {

struct EvaluationInput {
    std::filesystem::path image_path;
    std::string label;
};

inline std::string manifestCsvField(const std::string &value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) {
        return value;
    }
    std::string encoded = "\"";
    for (char ch : value) {
        if (ch == '\"') {
            encoded += '\"';
        }
        encoded += ch;
    }
    return encoded + '\"';
}

inline std::vector<std::string> parseManifestRow(const std::string &line, std::size_t line_number) {
    const auto fail = [line_number]() {
        throw std::runtime_error("Invalid manifest CSV at line " + std::to_string(line_number) +
                                 ": expected exactly two fields; quoted fields must fit on one line.");
    };
    std::vector<std::string> columns;
    std::size_t cursor = 0;
    for (;;) {
        std::string value;
        if (cursor < line.size() && line[cursor] == '\"') {
            ++cursor;
            bool closed = false;
            while (cursor < line.size()) {
                const char ch = line[cursor++];
                if (ch != '\"') {
                    value += ch;
                } else if (cursor < line.size() && line[cursor] == '\"') {
                    value += '\"';
                    ++cursor;
                } else {
                    closed = true;
                    break;
                }
            }
            if (!closed || (cursor < line.size() && line[cursor] != ',')) {
                fail();
            }
        } else {
            while (cursor < line.size() && line[cursor] != ',') {
                if (line[cursor] == '\"') {
                    fail();
                }
                value += line[cursor++];
            }
        }
        for (unsigned char ch : value) {
            if (std::iscntrl(ch)) {
                fail();
            }
        }
        columns.push_back(std::move(value));
        if (cursor == line.size()) {
            break;
        }
        ++cursor;
    }
    if (columns.size() != 2) {
        fail();
    }
    return columns;
}

inline std::vector<EvaluationInput> loadEvaluationManifest(const std::filesystem::path &manifest_path,
                                                          const std::filesystem::path &dataset_override) {
    namespace fs = std::filesystem;
    if (manifest_path.empty()) {
        throw std::runtime_error("Evaluation requires an explicit image,label manifest (--manifest or evaluation.manifest_path).");
    }
    std::ifstream input(manifest_path);
    if (!input.is_open()) {
        throw std::runtime_error("Cannot open evaluation manifest: " + manifest_path.string() +
                                 ". Supply --manifest with an explicit image,label CSV; labels are never inferred.");
    }

    const fs::path root = dataset_override.empty() ? fs::absolute(manifest_path).parent_path()
                                                   : fs::absolute(dataset_override);
    if (!dataset_override.empty() && !fs::is_directory(root)) {
        throw std::runtime_error("--dataset must name an existing directory: " + root.string());
    }
    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("Evaluation manifest is empty: " + manifest_path.string());
    }
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    if (line.compare(0, 3, "\xEF\xBB\xBF") == 0) {
        line.erase(0, 3);
    }
    const auto header = parseManifestRow(line, 1);
    if (header[0] != "image" || header[1] != "label") {
        throw std::runtime_error("Evaluation manifest header must be exactly image,label.");
    }

    const std::set<std::string> reserved_labels = {"NoDetection", "MissingImage", "UnreadableImage", "PipelineError"};
    std::set<fs::path> seen_paths;
    std::vector<EvaluationInput> rows;
    std::size_t line_number = 1;
    while (std::getline(input, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const auto columns = parseManifestRow(line, line_number);
        for (const std::string &column : columns) {
            if (column.empty() || column.find_first_not_of(" \t") == std::string::npos) {
                throw std::runtime_error("Empty image or label in evaluation manifest at line " + std::to_string(line_number));
            }
        }
        if (reserved_labels.count(columns[1]) != 0) {
            throw std::runtime_error("Reserved evaluation status used as a true label at line " + std::to_string(line_number) +
                                     ": " + columns[1]);
        }
        if (std::isspace(static_cast<unsigned char>(columns[1].front())) ||
            std::isspace(static_cast<unsigned char>(columns[1].back()))) {
            throw std::runtime_error("Evaluation labels must not contain leading or trailing whitespace.");
        }
        const fs::path image(columns[0]);
        const fs::path resolved = fs::weakly_canonical(image.is_absolute() ? image : root / image);
        if (!seen_paths.insert(resolved).second) {
            throw std::runtime_error("Duplicate image in evaluation manifest at line " + std::to_string(line_number) +
                                     ": " + resolved.string());
        }
        rows.push_back({resolved, columns[1]});
    }
    if (!input.eof()) {
        throw std::runtime_error("Failed while reading evaluation manifest: " + manifest_path.string());
    }
    if (rows.empty()) {
        throw std::runtime_error("Evaluation manifest must contain at least one labeled image.");
    }
    return rows;
}

inline void writeEvaluationManifestSnapshot(const std::filesystem::path &path, const std::vector<EvaluationInput> &rows) {
    std::ofstream output(path);
    output << "image,label\n";
    for (const EvaluationInput &row : rows) {
        output << manifestCsvField(row.image_path.string()) << ',' << manifestCsvField(row.label) << '\n';
    }
    output.flush();
    if (!output) {
        throw std::runtime_error("Failed to save evaluation manifest snapshot: " + path.string());
    }
}

}  // namespace vision::app
