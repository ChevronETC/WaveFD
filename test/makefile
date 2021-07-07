all		: \
	valgrind_prop2DAcoIsoDenQ_DEO2_FDTD \
	valgrind_prop2DAcoVTIDenQ_DEO2_FDTD \
	valgrind_prop2DAcoTTIDenQ_DEO2_FDTD \
	valgrind_prop3DAcoIsoDenQ_DEO2_FDTD \
	valgrind_prop3DAcoVTIDenQ_DEO2_FDTD \
	valgrind_prop3DAcoTTIDenQ_DEO2_FDTD 

valgrind_prop2DAcoIsoDenQ_DEO2_FDTD: valgrind_prop2DAcoIsoDenQ_DEO2_FDTD.cc
	g++ -I../src -g -o $@ $< -lfftw3f
	valgrind --leak-check=full ./$@

valgrind_prop2DAcoVTIDenQ_DEO2_FDTD: valgrind_prop2DAcoVTIDenQ_DEO2_FDTD.cc
	g++ -I../src -g -o $@ $< -lfftw3f
	valgrind --leak-check=full ./$@

valgrind_prop2DAcoTTIDenQ_DEO2_FDTD: valgrind_prop2DAcoTTIDenQ_DEO2_FDTD.cc
	g++ -I../src -g -o $@ $< -lfftw3f
	valgrind --leak-check=full ./$@

valgrind_prop3DAcoIsoDenQ_DEO2_FDTD: valgrind_prop3DAcoIsoDenQ_DEO2_FDTD.cc
	g++ -I../src -g -o $@ $< -lfftw3f
	valgrind --leak-check=full ./$@

valgrind_prop3DAcoVTIDenQ_DEO2_FDTD: valgrind_prop3DAcoVTIDenQ_DEO2_FDTD.cc
	g++ -I../src -g -o $@ $< -lfftw3f
	valgrind --leak-check=full ./$@

valgrind_prop3DAcoTTIDenQ_DEO2_FDTD: valgrind_prop3DAcoTTIDenQ_DEO2_FDTD.cc
	g++ -I../src -g -o $@ $< -lfftw3f
	valgrind --leak-check=full ./$@

make clean:
	/bin/rm -rf valgrind_prop2DAcoIsoDenQ_DEO2_FDTD
	/bin/rm -rf valgrind_prop2DAcoVTIDenQ_DEO2_FDTD
	/bin/rm -rf valgrind_prop2DAcoTTIDenQ_DEO2_FDTD
	/bin/rm -rf valgrind_prop3DAcoIsoDenQ_DEO2_FDTD
	/bin/rm -rf valgrind_prop3DAcoVTIDenQ_DEO2_FDTD
	/bin/rm -rf valgrind_prop3DAcoTTIDenQ_DEO2_FDTD
