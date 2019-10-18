FILE = stencil-computation
EXEC = exec
OUTPUT = openmp.out
FLAGS = -fopenmp -o

$(EXEC): $(FILE).c
	gcc $(FILE).c $(FLAGS) $(EXEC)

run: $(EXEC)
	@./$(EXEC) > $(OUTPUT)
	@echo "Output written to file \"$(OUTPUT)\""

clean:
	rm -rf $(EXEC) $(OUTPUT)