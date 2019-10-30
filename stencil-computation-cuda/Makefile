FILE = stencil-computation
EXEC = exec
OUTPUT = cuda.out
FLAGS = -o
$(EXEC): $(FILE).cu
	nvcc $(FILE).cu $(FLAGS) $(EXEC)

run: $(EXEC)
	@./$(EXEC) > $(OUTPUT)
	@echo "Output written to file \"$(OUTPUT)\""

clean:
	rm -rf $(EXEC) $(OUTPUT)