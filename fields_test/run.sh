


# Compile
nvcc -O3 -o fields_test/target/bin fields_test/fields_test.cu 
# Run if the compilation succeeds, if not, stop
if nvcc -O3 -o fields_test/target/bin fields_test/fields_test.cu; then
    ./fields_test/target/bin
else
    echo "Compilation failed."
    exit 1
fi
