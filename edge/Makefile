CC = g++
CFLAGS = -Wall -O2 -std=c++11

all: edge_detector

edge_detector: main.cpp DwtCusumEngine.cpp DwtCusumEngine.h
	$(CC) $(CFLAGS) -o edge_detector main.cpp DwtCusumEngine.cpp

clean:
	rm -f edge_detector
