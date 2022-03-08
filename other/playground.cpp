template<typename T>
class td;


int main()
{

	int i = 10;
	const int* const iPtr = &i;
	td<decltype(i)> iType;
	td<decltype(iPtr)> xType;
	auto t = i;



	decltype(x)

	return 0;
}