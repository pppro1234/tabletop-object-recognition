# Build and test

The native build requires a C++17 compiler, CMake 3.21 or newer, OpenCV
(core, imgproc, imgcodecs, highgui, videoio and dnn), and SQLite3 development
headers and libraries. Building the test suite also requires Python 3.10 or
newer; its evaluation contract tests use the Python standard library.

From the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure --output-junit "$PWD/build/test-results.xml"
```

CTest runs unit, integration, regression and performance tests. Run an
individual group with `ctest --test-dir build --output-on-failure -L unit`.
The build keeps C++ test assertions enabled in Release configurations.

Every test receives a separate temporary directory under
`build/test-tmp/<test-name>`. Test databases, images and metrics stay inside
that build tree, so separate build directories do not share test state.
The test helpers reset each test's generated state before using it.
Each CTest entry has a 120-second timeout.

The GitHub Actions workflow uses the same configure, build and test sequence.
It saves the JUnit report, CTest logs and generated performance metrics as
the `test-results` artifact, including on test failure. A local passing run
does not establish that a remote workflow has run.

The synthetic regression and performance fixtures are small, deterministic
checks for software regressions. Their accuracy and timing results are not
measurements of recognition quality or throughput on real camera input.
