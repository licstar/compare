#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <map>
#include <string>
#include <vector>
using namespace std;
struct node {
	string w1, w2;
	double val;
};
map<string, double *> dict;

const int MAX_STRING = 1000;
int size;
int ReadEmbedding(const char *file_name) {
	FILE *f = fopen(file_name, "rb");
	if (f == NULL) {
		printf("Embedding file not found\n");
		return -1;
	}
	int wordNum;
	fscanf(f, "%d", &wordNum);
	fscanf(f, "%d", &size);

	char str[MAX_STRING];
	double *tmp = new double[size];
	for (int b = 0; b < wordNum; b++) {
		char ch;
		fscanf(f, "%s%c", str, &ch);
		/*for (int i = 0; str[i]; i++){
		if (str[i] >= 'A' && str[i] <= 'Z'){
		str[i] = str[i] - 'A' + 'a';
		}
		}*/
		map<string, double*>::iterator it = dict.find(str);
		double *v = tmp;
		if (it != dict.end()) {
			if (it->second == NULL) {
				it->second = new double[size];
				v = it->second;
			}
		}
		for (int a = 0; a < size; a++)
			fscanf(f, "%lf", &v[a]);
	}
	fclose(f);
	return 0;
}

const double eps = 1e-8;
double cosvec(double *a, double *b) {
	if (a == NULL || b == NULL)
		return 0;
	double t1 = 0, t2 = 0, t3 = 0;
	for (int i = 0; i < size; i++) {
		t1 += a[i] * b[i];
		t2 += a[i] * a[i];
		t3 += b[i] * b[i];
	}
	return t1 / sqrt(t2 + eps) / sqrt(t3 + eps);
}


double pearson(vector<double> &a, vector<double> &b) {
	double avg_a = 0, avg_b = 0;
	int n = a.size();
	for (int i = 0; i < n; i++) {
		avg_a += a[i];
		avg_b += b[i];
	}
	avg_a /= n;
	avg_b /= n;
	double v1 = 0, v2 = 0, v3 = 0;
	for (int i = 0; i < n; i++) {
		v1 += (a[i] - avg_a) * (b[i] - avg_b);
		v2 += (a[i] - avg_a) * (a[i] - avg_a);
		v3 += (b[i] - avg_b) * (b[i] - avg_b);
	}
	return v1 / sqrt(v2 + eps) / sqrt(v3 + eps);
}

void solve(const char *dataset, const char *embedding) {
	vector<node> lst;

	char w1[MAX_STRING], w2[MAX_STRING];
	double val;
	FILE *fd = fopen(dataset, "r");
	while (fscanf(fd, "%s%s%lf", w1, w2, &val) != EOF) {
		node n;
		n.w1 = w1;
		n.w2 = w2;
		n.val = val;
		lst.push_back(n);
		if (embedding) {
			dict[n.w1] = NULL;
			dict[n.w2] = NULL;
		}

	}
	fclose(fd);

	if (embedding)
		ReadEmbedding(embedding);

	for (map<string, double*>::iterator it = dict.begin(); it != dict.end(); it++) {
		if (it->second == NULL) {
			fprintf(stderr, "cannot find word: %s\n", it->first.c_str());
		}
	}

	vector<double> aa, bb;
	for (int i = 0; i < lst.size(); i++) {
		double *v1 = dict[lst[i].w1];
		double *v2 = dict[lst[i].w2];
		aa.push_back(lst[i].val);
		bb.push_back(cosvec(v1, v2));
	}
	printf("%lf\n", pearson(aa, bb));
}


int main(int argc, char **argv) {
	if (argc != 2) {
		printf("Useage: ./ws embedding > pearson\n");
		return 0;
	}

	solve("ws353.txt", argv[1]);
	solve("ws353_relatedness.txt", NULL);
	solve("ws353_similarity.txt", NULL);

	return 0;
}
