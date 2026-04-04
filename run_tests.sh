#!/bin/bash
set -e

echo "=== Running ECHORA Test Suite ==="

for test_file in tests/test_*.py; do
    echo "Running $test_file..."
    PYTHONPATH=. python3 "$test_file"
done

echo "=== All Tests Passed Successfully ==="
