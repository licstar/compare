CC = g++
GCC = gcc
CFLAGS = -lm -O2 -Wall -funroll-loops -ffast-math
#CFLAGS = -lm -O2 -Wall

all: senna_tag

senna_tag : sennaseg.cpp
	$(CC) $(CFLAGS) $(OPT_DEF) sennaseg.cpp -fopenmp -DLINUX -o senna_tag

clean:
	rm -rf *.o senna_tag
