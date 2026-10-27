#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define FEATURES 5
#define MAX_ROWS 10000
#define LINE_LEN 512
#define EPOCHS 5000
#define LEARNING_RATE 0.05

typedef struct {
    double x[FEATURES];
    int label;
} Row;

static double weights[FEATURES] = {0};
static double bias = 0;

static double sigmoid(double z)
{
    if (z > 35.0) z = 35.0;
    if (z < -35.0) z = -35.0;
    return 1.0 / (1.0 + exp(-z));
}

/* CSV columns: label, then five numeric features. */
static int parse_row(char *line, Row *row)
{
    char *token = strtok(line, ",");
    if (!token) return 0;

    char *end;
    long label = strtol(token, &end, 10);
    if (*end != '\0' || (label != 0 && label != 1)) return 0;
    row->label = (int)label;

    for (int i = 0; i < FEATURES; i++) {
        token = strtok(NULL, ",");
        if (!token) return 0;

        row->x[i] = strtod(token, &end);
        if (end == token || (*end != '\0' && *end != '\r' && *end != '\n'))
            return 0;
    }

    return 1;
}

static int load_csv(const char *filename, Row *rows, int capacity)
{
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror(filename);
        return -1;
    }

    char line[LINE_LEN];
    int count = 0;

    /* First line is the CSV header. */
    if (!fgets(line, sizeof line, file)) {
        fclose(file);
        return 0;
    }

    while (count < capacity && fgets(line, sizeof line, file)) {
        Row row;
        if (parse_row(line, &row))
            rows[count++] = row;
    }

    fclose(file);
    return count;
}

static double predict_probability(const Row *row)
{
    double z = bias;
    for (int i = 0; i < FEATURES; i++)
        z += weights[i] * row->x[i];

    return sigmoid(z);
}

static void train(const Row *rows, int count)
{
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double gradient[FEATURES] = {0};
        double bias_gradient = 0;

        for (int r = 0; r < count; r++) {
            double error = predict_probability(&rows[r]) - rows[r].label;

            for (int i = 0; i < FEATURES; i++)
                gradient[i] += error * rows[r].x[i];

            bias_gradient += error;
        }

        for (int i = 0; i < FEATURES; i++)
            weights[i] -= LEARNING_RATE * gradient[i] / count;

        bias -= LEARNING_RATE * bias_gradient / count;
    }
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s customers.csv\n", argv[0]);
        return 1;
    }

    static Row rows[MAX_ROWS];
    int count = load_csv(argv[1], rows, MAX_ROWS);

    if (count < 0) return 1;
    if (count < 2) {
        fprintf(stderr, "Need at least two valid data rows.\n");
        return 1;
    }

    train(rows, count);

    printf("Trained on %d labeled records.\n\n", count);

    for (int r = 0; r < count; r++) {
        double probability = predict_probability(&rows[r]);
        printf("Row %d: label=%d, theft score=%.1f%%, %s\n",
               r + 1, rows[r].label, probability * 100.0,
               probability >= 0.5 ? "flag for review" : "not flagged");
    }

    return 0;
}
