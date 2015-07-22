CC = g++
GCC = gcc
CFLAGS = -lm -O2 -Wall -funroll-loops -ffast-math
#CFLAGS = -lm -O2 -Wall

all: cnn_senna


cnn_senna : cnn_senna.cpp
	$(CC) $(CFLAGS) $(OPT_DEF) cnn_senna.cpp -DLINUX -fopenmp -O2 -g -std=c++0x -o cnn_senna

clean:
	rm -rf *.o cnn_senna 
