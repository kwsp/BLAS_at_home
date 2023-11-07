#!/bin/bash
set -e

# Build with CMake
cmake --preset=clang-with-warnings -B build
cmake --build build -j

# Link compile_commands.json
if [[ -f compile_commands.json ]]; then
  rm compile_commands.json
fi
ln -s ./build/compile_commands.json compile_commands.json

# Run gtest
echo
./build/bah_test
