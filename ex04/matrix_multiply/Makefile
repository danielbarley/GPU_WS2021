######
# C++

SRC_DIR = src
OBJ_DIR = obj

SRC = $(wildcard $(SRC_DIR)/*.cpp)
OBJ = $(SRC:$(SRC_DIR)/%.cpp=%.out)

CC = g++ --std=c++11 -fdiagnostics-color=always -O2
CPPFLAGS += -Iinclude -Ilibs/argparse
CFLAGS += -Wall -Wextra -Wpedantic -fopenmp
LDFLAGS += -Llib
LDLIBS += -lpthread

.PHONY: all clean

all: $(OBJ)

%.out: $(SRC_DIR)/%.cpp
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

clean:
	$(RM) $(OBJ)
