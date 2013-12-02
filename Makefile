CXX=/usr/bin/g++

OPENCVINCL=`pkg-config opencv --cflags`
OPENCVLIBS=`pkg-config opencv --cflags --libs`

CFLAGS =  -I./newmat/ -I/usr/local/include/ -I/opt/local/include/ 
LDFLAGS = -L/usr/local/lib -L/opt/local/lib   -lsiftfast 


OBJECTS = newmat/libnewmat.a\
	steer_kern.o\
	tvg_energy.o\
	tvg_descriptor.o\
	example.o

all:
	./buildNewmat.sh;
	make tvg2D;

%.o: %.cpp  tvg2D.h
	$(CXX) -O3 $(OPENCVINCL) $(CFLAGS) -c $*.cpp

tvg2D: $(OBJECTS)  tvg2D.h 
	$(CXX) -O3 -o tvg_example $(OBJECTS) $(OPENCVLIBS) $(LDFLAGS)

clean:
	/bin/rm -rf *.o
	/bin/rm -rf *~
	/bin/rm -rf tvg_example

