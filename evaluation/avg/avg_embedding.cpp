#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <vector>
#include <sstream>
//#include <omp.h>
#ifdef LINUX
#include <sys/time.h>
#else
#include <time.h>
#endif

using namespace std;
typedef vector<int> doc;



#include <string>
#include <map>
#include <stdio.h>
#include <vector>
#include <unordered_map>
using namespace std;

/*
1. 读入embedding（字词的在一个文件里），放到hash表中。
2. 读入训练集、测试集（自动分出验证集？），并放入合理的数据结构中。
训练集、测试集、验证集 vec<doc>
doc=vec<word>

*/

#define MAX_STRING 10000


map<string, int> dict; //字典中所有的字 str->id
vector<string> vocab; //所有的词、字，按照顺序存在这里。 id->str

struct embedding_t {
	int size; //里面包含多少个变量（value 里面的变量个数） size = element_size * element_num
	int element_size; //一个向量的长度
	int element_num; //向量的个数
	double *value; //所有的参数

	void init(int element_size, int element_num) {
		this->element_size = element_size;
		this->element_num = element_num;
		size = element_size * element_num;
		value = new double[size];
	}
};

void ReadWord(char *word, FILE *fin) {
	int a = 0, ch;

	while (!feof(fin)) {
		ch = fgetc(fin);

		if (ch == 13) continue;

		if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
			if (a > 0) {
				if (ch == '\n') ungetc(ch, fin);
				break;
			}

			if (ch == '\n') {
				strcpy(word, (char *)"</s>");
				return;
			} else continue;
		}

		word[a] = ch;
		a++;

		if (a >= MAX_STRING) {
			printf("Too long word found!\n");   //truncate too long words
			a--;
		}
	}
	word[a] = 0;
}

int ReadWordIndex(FILE *fin) {
	char word[MAX_STRING];
	ReadWord(word, fin);
	if (feof(fin)) return -1;
	if (strcmp(word, "</s>") == 0)
		return -2;

	map<string, int>::iterator it = dict.find(word);
	if (it == dict.end()) {
		return -1;
	} else {
		return it->second;
	}
}

int ReadEmbedding(embedding_t &words, char *file_name) {
	FILE *f = fopen(file_name, "rb");
	if (f == NULL) {
		printf("Embedding file not found\n");
		return -1;
	}
	int wordNum, size;
	fscanf(f, "%d", &wordNum);
	fscanf(f, "%d", &size);

	words.init(size, wordNum);

	char str[MAX_STRING];
	for (int b = 0; b < wordNum; b++) {
		char ch;
		fscanf(f, "%s%c", str, &ch);
		string s = str;
		dict[s] = vocab.size();
		vocab.push_back(s);
		for (int a = 0; a < size; a++)
			fscanf(f, "%lf", &words.value[a + b * size]);
		/*
		double len = 0;
		for (int a = 0; a < size; a++)
		len += M[a + b * size] * M[a + b * size];
		len = sqrt(len);
		for (int a = 0; a < size; a++)
		M[a + b * size] /= len;
		*/
	}
	fclose(f);
	return 0;
}

/*
读取文档，可能是训练集、测试集。
每行一篇文档，其中第一个数字是类别信息。后面是原文。每个词都用 字~字 的形式来表示。
*/
void ReadDocs(const char *file_name, vector<doc> &docs, vector<int> &labels, const char *dataset) {
	FILE *fi = fopen(file_name, "rb");
	char label[MAX_STRING];
	int unknown = 0;
	int total = 0;
	while (1) {
		if (feof(fi)) break;
		doc d;
		//第一个是标签，先读入标签
		ReadWord(label, fi);
		if (feof(fi)) break;
		int lb = atoi(label);
		labels.push_back(lb);

		while (1) { //开始读全文信息
			int word = ReadWordIndex(fi);
			if (feof(fi)) break;
			if (word == -1) {
				unknown++; //低频词
				continue;
			}
			//word_count++;
			if (word == -2) break; //一行结束了
			d.push_back(word);
		}
		if (d.size() > 0) {
			docs.push_back(d);
			total += d.size();
		}
	}
	printf("%s data: N(docs):%d, words:%d, unknown(ignore):%d\n", dataset, (int)docs.size(), total, unknown);
}

//==========================================================


double nextDouble(double s) {
	return rand() / (RAND_MAX + 1.0) * s - s / 2;
}


const int H = 500; //隐藏层，就是词向量的大小
const int MAX_C = 50; //最大分类数
const int MAX_F = 1000; //输入层最大的大小
const char *model_name = "model_idf";

//const char *train_file = "train.txt";
const char *train_file = "E:\\ecnn数据\\复旦新闻\\train.txt";
const char *valid_file = "valid.txt";
const char *test_file = "E:\\ecnn数据\\复旦新闻\\test.txt";

int class_size; //分类数
int vector_size; //一个词单元的向量大小 = 词向量大小（约50） + 所有特征的大小（约10）

//===================== 所有要优化的参数 =====================

embedding_t words; //词向量

double *A; //特征矩阵：[分类数][隐藏层] 第二层的权重
double biasOutput[H]; //TODO class_size

//===================== 已知数据 =====================

//训练集
vector<doc> data; //训练数据：[样本数][特征数]
vector<vector<double> > _data; //训练数据：[样本数][特征数]

//int N; //训练集大小
//int uN; //未知词
vector<int> b; //目标矩阵[样本数] 训练集

//验证集
vector<doc> vdata; //测试数据：[样本数][特征数]
vector<vector<double> > _vdata; //测试数据：[样本数][特征数]
//int vN; //测试集大小
//int uvN; //未知词
vector<int> vb; //目标矩阵[样本数] 测试集

//测试集
vector<doc> tdata; //测试数据：[样本数][特征数]
vector<vector<double> > _tdata; //测试数据：[样本数][特征数]
//int tN; //测试集大小
//int utN; //未知词
vector<int> tb; //目标矩阵[样本数] 测试集





double time_start;
double lambda = 0;//0.01; //正则项参数权重
double alpha = 0.01; //学习速率
int iter = 0;

double getTime() {
#ifdef LINUX
	timeval tv;
	gettimeofday(&tv, 0);
	return tv.tv_sec + tv.tv_usec * 1e-6;
#else
	return clock();
#endif
}

double nextDouble() {
	return rand() / (RAND_MAX + 1.0);
}

void writeFile(const char *name, double *A, int size) {
	FILE *fout = fopen(name, "wb");
	fwrite(A, sizeof(double), size, fout);
	fclose(fout);
}


int readFile(const char *name, double *A, int size) {
	FILE *fin = fopen(name, "rb");
	if (!fin)
		return 0;
	int len = (int)fread(A, sizeof(double), size, fin);
	fclose(fin);
	return len;
}


/*
训练集、测试集、验证集 vec<doc>
doc=vec<word>
word=(string)ch~ch~ch

参数
embedding 每个词、字都有一个和某个hash表对应

评测 doc，返回分类（保留中间变量，用来算梯度）

1. 读取每个集合 训练集、测试集、验证集
2. 每篇文档求一个平均值，存在vector<double>里
3. 训练，实时输出训练结果
4. 参数不用embedding初始化也看看效果
5. 做一个bag of words的实验，用liblinear
*/


//对一个集合里面的所有文档求一个embedding的平均值
void ConvertData(vector<doc> &data, vector<vector<double> > &_data) {
	for (size_t i = 0; i < data.size(); i++) {
		doc &d = data[i];
		double v[H] = { 0 };
		double w = 0; //权重
		for (size_t j = 0; j < d.size(); j++) {
			double weight = 1;
			w += weight;
			for (int k = 0; k < words.element_size; k++) {
				v[k] += weight * words.value[d[j] * words.element_size + k];
			}
		}
		vector<double> vv;
		vv.reserve(words.element_size);
		for (int k = 0; k < words.element_size; k++) {
			vv.push_back(v[k] / w);
		}
		_data.push_back(vv);
	}
}

void WriteData(const char *file, vector<vector<double> > &data, vector<int> &b) {
	//char fname[100];
	//sprintf(fname, "%s_l", file);

	//	test_file = argv[4];
	FILE *fout = fopen(file, "w");
	for (int i = 0; i < data.size(); i++) {
		fprintf(fout, "%d", b[i]);
		for (int j = 0; j < data[i].size(); j++) {
			fprintf(fout, " %d:%.16lf", j + 1, data[i][j]);
		}
		fprintf(fout, "\n");
	}

}

int main(int argc, char **argv) {
	if (argc != 6) {
		printf("Useage: ./avg_embedding embedding train test train_out test_out\n");
		return 0;
	}

	printf("read embedding\n");

	train_file = argv[2];
	test_file = argv[3];

	if (ReadEmbedding(words, argv[1]) != -1) {
		printf("initialized with %s\n", argv[1]);
		/*double sum = 0;
		for (int i = 0; i < words.size; i++) {
			sum += words.value[i] * words.value[i];
		}
		sum = sqrt(sum / words.size * 12);
		for (int i = 0; i < words.size; i++) {
			words.value[i] /= sum;
		}*/

		//for (int i = 0; i < words.size; i++) {
		//	words.value[i] = (nextDouble() - 0.5);
		//}
	} else {
		printf("not initialized\n");
	}

	printf("read data\n");
	ReadDocs(train_file, data, b, "Train");
	ReadDocs(test_file, tdata, tb, "Test");

	ConvertData(data, _data);
	ConvertData(tdata, _tdata);

	WriteData(argv[4], _data, b);
	WriteData(argv[5], _tdata, tb);

	return 0;
}
