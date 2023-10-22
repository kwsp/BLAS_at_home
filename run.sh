set -e

cmake -B build
cmake --build build -j

if [[ -f compile_commands.json ]]; then
  rm compile_commands.json
fi

ln -s ./build/compile_commands.json compile_commands.json

echo
./build/bah_test
