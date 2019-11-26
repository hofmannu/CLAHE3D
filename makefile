CC = g++
CFLAGS = -Wall -Wpedantic -ggdb
LFLAGS = -Wall -Wpedantic -ggdb
LIBS = 

obj/histeq.o:
	$(CC) $(CFLAGS) -c src/histeq.cpp -o obj/histeq.o

obj/interpGrid.o:
	$(CC) $(CFLAGS) -c src/interpGrid.cpp -o obj/interpGrid.o

obj/main.o: obj/histeq.o obj/interpGrid.o
	$(CC) $(CFLAGS) -c src/main.cpp -o obj/main.o

main: obj/main.o
	$(CC) $(LFLAGS) -o bin/main obj/histeq.o obj/interpGrid.o obj/main.o $(LIBS)

clean:
	rm -f obj/*.o

.PHONY: obj/histeq.o obj/main.o obj/interpGrid.o
