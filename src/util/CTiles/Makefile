.PHONY : tiles

tiles: tiles.so

tilesInt.o: tilesInt.C
	c++ -fno-strict-aliasing -Wno-long-double -no-cpp-precomp -fno-common -dynamic -DNDEBUG -g -O3 -Wall -Wstrict-prototypes -I/Users/pvrancx/anaconda/include/python2.7 -c tilesInt.C -o tilesInt.o

.omic_lookup: tile.cpp
	c++ -fno-strict-aliasing -Wno-long-double -no-cpp-precomp -fno-common -dynamic -DNDEBUG -g -O3 -Wall -Wstrict-prototypes -I/Users/pvrancx/anaconda/include/python2.7 -c tiles.cpp -o tiles.o
tiles.so: tiles.o tilesInt.o
	c++ -Wl,-F. -Wl,-F. -Wl,-undefined -Wl,dynamic_lookup -L/Users/pvrancx/anaconda/lib/ -ldl -lpython2.7  tiles.o tilesInt.o -o tiles.so
