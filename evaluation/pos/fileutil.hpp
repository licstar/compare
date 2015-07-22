#include <string>
#include <map>
#include <vector>
#include <algorithm>
using namespace std;

#define MAX_STRING 1000


map<string, int> dict; //字典中所有的字 str->id
vector<string> vocab; //所有的词、字，按照顺序存在这里。 id->str
int ReadEmbedding(embedding_t &words, const char *file_name) {
	FILE *f = fopen(file_name, "rb");
	if (f == NULL) {
		printf("Embedding file not found\n");
		return -1;
	}
	int wordNum, size;
	fscanf(f, "%d", &wordNum);
	fscanf(f, "%d", &size);

	words.init(size, wordNum + 2);

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
	for (int b = wordNum; b < wordNum + 2; b++)
	for (int a = 0; a < size; a++)
		words.value[a + b * size] = rand() / (RAND_MAX + 1.0) - 0.5;
	return 0;
}


int padding_id = 0;
int unknown_id = 0;


void readWord(char *word, FILE *fin) {
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


data_t addWord(char *word) {
	data_t ret;
	string w = word;


	map<string, int>::iterator it = dict.find(w);
	if (it != dict.end()) {
		ret.word = it->second;
	} else {
		ret.word = dict[w] = vocab.size();
		vocab.push_back(w);
	}
	//ret.ch = addRealWords(word);
	return ret;
}

data_t getWord(char *word) {
	data_t ret;
	string w = word;
	map<string, int>::iterator it = dict.find(w);
	if (it != dict.end()) {
		ret.word = it->second;
	} else {
		ret.word = unknown_id;
	}
	//ret.ch = addRealWords(word);
	return ret;
}

data_t readWordIndex(FILE *fin, int &tag, bool add = false) {
	char word[MAX_STRING];
	data_t ret;
	ret.word = padding_id;

	readWord(word, fin);
	if (feof(fin)) return ret;

	tag = 0;
	if (strcmp("</s>", word) != 0) {
		for (int k = strlen(word) - 1; k >= 0; k--) {
			if (word[k] == '/') {
				tag = atoi(word + k + 1);
				word[k] = 0;
				return getWord(word);
				break;
			}
		}
	} else {
		ret.word = padding_id;
	}
	return ret;
}

void learnVocab(const char *train_file) {
	FILE *fi = fopen(train_file, "rb");
	while (1) {
		int tag;
		readWordIndex(fi, tag, true);
		if (feof(fi)) break;
	}
	fclose(fi);
}

//
//void init(const char *dict_file){
//	/*chk["</s>"] = 0;
//	chk["unknown"] = 1;
//	chk["padding"] = 2;
//	chk["number"] = 3;
//	chk["letter"] = 4;
//	chk["numletter"] = 5;
//	*/
//	char ch[100];
//	FILE *fin = fopen(dict_file, "r");
//	while(fgets(ch, sizeof(ch), fin)){
//		int len = strlen(ch);
//		while((ch[len-1] == '\r' || ch[len-1] == '\n') && len > 0){
//			ch[len-1] = 0;
//			len--;
//		}
//		len = chk.size();
//		chk[ch] = len;
//	}
//	fclose(fin);
//}



void init(const char *embedding_file) {
	ReadEmbedding(words, embedding_file);
	padding_id = words.element_num - 2;
	unknown_id = words.element_num - 1;
}

int lineMax = 0;
void readAllData(const char *file, const char *dataset, int window_size, data_t *&data, int *&b, int &N, int &uN) {
	vector<vector<pair<data_t, int> > > mydata;
	FILE *fi = fopen(file, "rb");

	vector<pair<data_t, int> > line;

	data_t padding; //这个想办法初始化一下
	padding.word = padding_id;
	//	padding.ch = NULL;
	int lines = 0;
	N = 0;
	while (1) {
		int tag;
		data_t dt = readWordIndex(fi, tag);
		if (feof(fi)) break;
		line.push_back(make_pair(dt, tag));

		if (dt.word == padding_id) {
			line.pop_back();
			mydata.push_back(line);
			N += line.size();
			line.clear();
			lines++;
		}
	}
	fclose(fi);
	
	data = new data_t[N * window_size];
	b = new int[N];

	int hw = (window_size - 1) / 2;
	//int hw = 2;

	int unknown = 0;
	for (size_t i = 0, offset = 0; i < mydata.size(); i++) {
		vector<pair<data_t, int> > &vec = mydata[i];
		lineMax = max(lineMax, (int)vec.size());
		for (int j = 0; j < (int)vec.size(); j++, offset += window_size) {
			for (int k = hw; k > 0; k--) {
				if (j - k >= 0) {
					data[offset + hw - k] = vec[j - k].first;
				} else {
					data[offset + hw - k] = padding; //PADDING
				}
			}
			for (int k = 1; k <= hw; k++) {
				if (j + k < (int)vec.size()) {
					data[offset + hw + k] = vec[j + k].first;
				} else {
					data[offset + hw + k] = padding; //PADDING
				}
			}
			data[offset + hw] = vec[j].first;
			b[offset / window_size] = vec[j].second;
			if (vec[j].first.word == unknown_id) {
				//printf("%s\t", vec[j].first.ch);
				unknown++;
			}
		}
	}

	printf("%s data: N(words):%d, unknown:%d, lines: %d\n", dataset, N, unknown, lines);
	uN = unknown;

}