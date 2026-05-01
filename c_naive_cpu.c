#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include<string.h>
#define _GNU_SOURCE

typedef struct
{
    double data_loading;
    double fwd_matmul1;
    double fwd_bias1;
    double fwd_relu;
    double fwd_matmul2;
    double fwd_bias2;
    double fwd_softmax;
    double cross_entropy;
    double bwd_output_grad;
    double bwd_matmul2;
    double bwd_bias2;
    double bwd_relu;
    double bwd_matmul1;
    double bwd_bias1;
    double weight_updates;
    double total_time;
}TimingStats;

//Helper function to get time difference in seconds
double get_time_diff(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define BATCH_SIZE 4
#define TRAIN_SIZE 10000
#define TEST_SIZE 1000
#define EPOCHS 3
#define LEARNING_RATE 0.01

typedef struct
{
    float *weight1;
    float *weight2;
    float *bias1;
    float *bias2;   
    float *grad_weight1;
    float *grad_weight2;
    float *grad_bias1;
    float *grad_bias2;
}NeuralNetwork;

//Load batched img data
void load_data(const char *filename, float *data, int size){
    FILE *file = fopen(filename, "rb");
    if(file == NULL){
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    size_t read_size = fread(data, sizeof(float), size, file);
    if(read_size != size){
        fprintf(stderr, "Error reading data: expected %d elements, got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);
}

//load batched label data
void load_labels(const char *filename, int *labels, int size){
    FILE *file = fopen(filename, "rb");
    if(file == NULL){
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    size_t read_size = fread(labels, sizeof(int), size, file);
    if(read_size != size){
        fprintf(stderr, "Error reading labels: expected %d elements, got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);
}

//optimal uniform He init for ReLU activations
void initialize_weights(float *weight, int input_size, int output_size) {
    float scale = sqrtf(2.0/input_size);
    for(int i = 0; i < input_size * output_size; i++) {
        weight[i] = ((float) rand() / RAND_MAX) * 2 * scale - scale; // Uniform distribution in [-scale, scale] 
    }
}

//basic init for bias 
void initialize_bias(float *bias, int size){
    for(int i=0;i<size;i++){
        bias[i] = 0.0f;
    }
}

//normalize dataset using mnsit mean and std
void normalize_data(float *data, int size){
    const float mean = 0.1307f;
    const float std = 0.3081f;
    for(int i=0;i<size;i++){
        data[i] = (data[i] - mean) / std;
    }
}

//softmax function for output layer that works with batched inputs
void softmax(float *x, int batch_size, int size){
    for(int b=0;b<batch_size;b++){
        float max = x[b*size];
        for(int i=1;i<size;i++){
            if(x[b*size+i] > max){
                max = x[b*size+i];
            }
        } 
        float sum = 0.0f;
        for(int i=0;i<size;i++){
            x[b*size+i] = expf(x[b*size+i] - max); // Subtract max for numerical stability
            sum += x[b*size+i];
        }
        for(int i=0;i<size;i++){
            x[b*size+i] = fmaxf(x[b*size+i] / sum, 1e-7f); // Makes sure no probability is exactly zero
        }
    }
}

/*
Row major order matrix multiplication C = A @ B where A is m x n, B is n x k, and C is m x k
(C-style) A[i][j] → A[i * num_cols + j] //row no times lenghth of row gives the stride to the start of 
the row, plus the column index gives the offset within the row
C[i][j] = Σ A[i][l] * B[l][j]
*/

void matmul_a_b(float *A,float *B,float *C,int m,int n,int k){
    for(int i=0;i<m;i++){
        for(int j=0;j<k;j++){
            C[i*k+j] = 0.0f;
            for(int l=0;l<n;l++){
                C[i*k+j] += A[i*n+l] * B[l*k+j];
            }
        }
    }
}

//Matrix multplication AT @ B
/* Shapes:
A: (m × n)
Aᵀ: (n × m)
B: (m × k)
C: (n × k)
Original A:
A[row][col] = A[l][i]
But in Aᵀ:
Aᵀ[i][l] = A[l][i]
So: A[l * n + i] = Aᵀ[i][l]
*/

void matmul_at_b(float *A, float *B, float *C, int m, int n, int k) {
    for (int i = 0; i < n; i++) { // rows of Aᵀ
        for (int j = 0; j < k; j++) { // cols of B
            C[i * k + j] = 0.0f;
            for (int l = 0; l < m; l++) {  // shared dimension
                C[i * k + j] += A[l * n + i] * B[l * k + j]; 
            }
        }
    }
}

// Matrix multiplication A @ B.T
/* Shapes:
A: (m × n)
B: (k × n)
Bᵀ: (n × k)
C: (m × k)
*/
void matmul_a_bt(float *A, float *B, float *C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            C[i * k + j] = 0.0f;
            for (int l = 0; l < n; l++) {
                C[i * k + j] += A[i * n + l] * B[j * n + l];
            }
        }
    }
}

//Relu activation function for forward pass
void relu_forward(float *x, int size){
    for(int i=0;i<size;i++){
        x[i] = fmaxf(0.0f,x[i]);
    }
}

//Add bias
void bias_forward(float *x, float *bias, int batch_size, int size){
    for(int b=0;b<batch_size;b++){
        for(int i=0;i<size;i++){
            x[b*size+i] += bias[i];
        }
    }
}

//forward function with timing pipelime
/*input
 → matmul (W1)
 → bias1
 → ReLU
 → matmul (W2)
 → bias2
 → softmax
 */
void forward_timed(NeuralNetwork *nn, float *input, float *hidden, float *output, int batch_size, TimingStats *stats) {
    struct timespec start, end;
    //Equivalent pytorch code: x → fc1 → +bias → ReLU → fc2 → +bias → softmax
    // Input to Hidden (X @ W1)
    clock_gettime(CLOCK_MONOTONIC, &start); //compile with -lrt flag to use clock_gettime
    matmul_a_b(input, nn->weight1, hidden, batch_size, INPUT_SIZE, HIDDEN_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_matmul1 += get_time_diff(start, end);
    
    // Add bias1
    clock_gettime(CLOCK_MONOTONIC, &start);
    bias_forward(hidden, nn->bias1, batch_size, HIDDEN_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_bias1 += get_time_diff(start, end);
    
    // Apply ReLU
    clock_gettime(CLOCK_MONOTONIC, &start);
    relu_forward(hidden, batch_size * HIDDEN_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_relu += get_time_diff(start, end);

    // Hidden to Output (Hidden @ W2)
    clock_gettime(CLOCK_MONOTONIC, &start);
    matmul_a_b(hidden, nn->weight2, output, batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_matmul2 += get_time_diff(start, end);

    // Add bias2
    clock_gettime(CLOCK_MONOTONIC, &start);
    bias_forward(output, nn->bias2, batch_size, OUTPUT_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_bias2 += get_time_diff(start, end);
    
    // Apply softmax
    clock_gettime(CLOCK_MONOTONIC, &start);
    softmax(output, batch_size, OUTPUT_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->fwd_softmax += get_time_diff(start, end);
}

/*Cross entropy loss function that works with batches
1. Take predicted probabilities
2. Pick correct class probability
3. Measure how “surprised” we are using log
4. Average over batch
✔ penalizes wrong predictions heavily
✔ smooth gradients
✔ stable training
✔ aligns with probability theory
*/
float cross_entropy_loss(float *output, int *labels, int batch_size) {
    float total_loss = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        total_loss -= logf(fmaxf(output[b * OUTPUT_SIZE + labels[b]], 1e-7f)); //Uses fmaxf to avoid log(0)
    }
    return total_loss / batch_size; //Averages over batch
}

// Zero out gradients
void zero_grad(float *grad, int size) {
    memset(grad, 0, size * sizeof(float));
}

// ReLU backward
void relu_backward(float *grad, float *x, int size) {
    for (int i = 0; i < size; i++) {
        grad[i] *= (x[i] > 0);
    }
}

// Bias backward: We sum gradients across all samples (rows) for each neuron (column) to get bias gradients
void bias_backward(float *grad_bias, float *grad, int batch_size, int size) {
    for (int i = 0; i < size; i++) {
        grad_bias[i] = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            grad_bias[i] += grad[b * size + i]; //∂L/∂b[i] = Σ over batch ∂L/∂z[b][i]
        }
    }
}

// Compute gradients for output layer
void compute_output_gradients(float *grad_output, float *output, int *labels, int batch_size) {
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            grad_output[b * OUTPUT_SIZE + i] = output[b * OUTPUT_SIZE + i]; //Copy softmax probabilities into grad_output
        }
        grad_output[b * OUTPUT_SIZE + labels[b]] -= 1.0f;   // ∂L/∂z[b][i] = p[i] - 1: Subtract 1 from the correct class → (p - y_one_hot)
    }
    // Divide gradients by batch_size to match PyTorch behavior
    for (int i = 0; i < batch_size * OUTPUT_SIZE; i++) {
        grad_output[i] /= batch_size;
    }
}

/* Compute gradients for weights and biases
grad_weights = ∂L/∂W
grad_bias    = ∂L/∂b
(predicted - actual) → grad_output
xᵀ @ grad_output → grad_weights. This function is literally doing: manual version of xᵀ @ grad_output
 */
void update_gradients(float *grad_weights, float *grad_bias, float *grad_layer, float *prev_layer, int batch_size, int prev_size, int curr_size) {
    for (int i = 0; i < curr_size; i++) { // Loop over each neuron in current layer
        for (int j = 0; j < prev_size; j++) {// Compute gradients for weights connecting prev_layer → current neuron i
            for (int b = 0; b < batch_size; b++) { // Accumulate gradient over batch                                               
                grad_weights[i * prev_size + j] += grad_layer[b * curr_size + i] * prev_layer[b * prev_size + j]; // ∂L/∂W[i][j] = Σ_b (grad_layer[b][i] * prev_layer[b][j])
            }
        }
        // Compute bias gradient for neuron i
        // ∂L/∂b[i] = Σ_b grad_layer[b][i]
        for (int b = 0; b < batch_size; b++) {
            grad_bias[i] += grad_layer[b * curr_size + i];
        }
    }
}

// Backward pass function with timing
void backward_timed(NeuralNetwork *nn, float *input, float *hidden, float *output, int *labels, int batch_size, TimingStats *stats) {
    struct timespec start, end;
    
    // Initialize gradients to zero
    zero_grad(nn->grad_weight1, HIDDEN_SIZE * INPUT_SIZE);
    zero_grad(nn->grad_weight2, OUTPUT_SIZE * HIDDEN_SIZE);
    zero_grad(nn->grad_bias1, HIDDEN_SIZE);
    zero_grad(nn->grad_bias2, OUTPUT_SIZE);

    // Compute gradients for output layer
    clock_gettime(CLOCK_MONOTONIC, &start);
    float *grad_output = malloc(batch_size * OUTPUT_SIZE * sizeof(float));
    compute_output_gradients(grad_output, output, labels, batch_size);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->bwd_output_grad += get_time_diff(start, end);

    // Update gradients for weights2 (W2.grad = grad_output.T @ hidden)
    clock_gettime(CLOCK_MONOTONIC, &start);
    matmul_at_b(hidden, grad_output, nn->grad_weight2, batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->bwd_matmul2 += get_time_diff(start, end);

    // Update gradients for bias2
    clock_gettime(CLOCK_MONOTONIC, &start);
    bias_backward(nn->grad_bias2, grad_output, batch_size, OUTPUT_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->bwd_bias2 += get_time_diff(start, end);

    // Compute dX2 (gradient of loss w.r.t. input of second layer)
    float *dX2 = malloc(batch_size * HIDDEN_SIZE * sizeof(float));

    // grad_output @ W2.T = dX2 -> (B, 10) @ (10, 256) = (B, 256)
    matmul_a_bt(grad_output, nn->weight2, dX2, batch_size, OUTPUT_SIZE, HIDDEN_SIZE);

    // Compute d_ReLU_out (element-wise multiplication with ReLU derivative)
    clock_gettime(CLOCK_MONOTONIC, &start);
    float *d_ReLU_out = malloc(batch_size * HIDDEN_SIZE * sizeof(float));
    for (int i = 0; i < batch_size * HIDDEN_SIZE; i++) {
        d_ReLU_out[i] = dX2[i] * (hidden[i] > 0);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->bwd_relu += get_time_diff(start, end);
    
    // Update gradients for weights1 (W1.grad = d_ReLU_out.T @ input)
    clock_gettime(CLOCK_MONOTONIC, &start);
    matmul_at_b(input, d_ReLU_out, nn->grad_weight1, batch_size, INPUT_SIZE, HIDDEN_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->bwd_matmul1 += get_time_diff(start, end);

    // Update gradients for bias1
    clock_gettime(CLOCK_MONOTONIC, &start);
    bias_backward(nn->grad_bias1, d_ReLU_out, batch_size, HIDDEN_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->bwd_bias1 += get_time_diff(start, end);

    // Free allocated memory
    free(grad_output);
    free(dX2);
    free(d_ReLU_out);
}

// gradient descent step with timing
void update_weights_timed(NeuralNetwork *nn, TimingStats *stats) {
    struct timespec start, end;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++) {
        nn->weight1[i] -= LEARNING_RATE * nn->grad_weight1[i];
    }
    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++) {
        nn->weight2[i] -= LEARNING_RATE * nn->grad_weight2[i];
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        nn->bias1[i] -= LEARNING_RATE * nn->grad_bias1[i];
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        nn->bias2[i] -= LEARNING_RATE * nn->grad_bias2[i];
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    stats->weight_updates += get_time_diff(start, end);
}

// Train function with comprehensive timing
void train_timed(NeuralNetwork *nn, float *X_train, int *y_train) {
    float *hidden = malloc(BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    float *output = malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));

    int num_batches = TRAIN_SIZE / BATCH_SIZE;
    
    // Initialize timing stats
    TimingStats stats = {0};
    
    // Start total timing
    struct timespec total_start, total_end, step_start, step_end;
    clock_gettime(CLOCK_MONOTONIC, &total_start);

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * BATCH_SIZE;
            
            // Data loading timing (minimal since data is already in memory)
            clock_gettime(CLOCK_MONOTONIC, &step_start);
            float *batch_input = &X_train[start_idx * INPUT_SIZE];
            int *batch_labels = &y_train[start_idx];
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            stats.data_loading += get_time_diff(step_start, step_end);
            
            // Forward pass with timing
            forward_timed(nn, batch_input, hidden, output, BATCH_SIZE, &stats);

            // Cross entropy loss timing  
            clock_gettime(CLOCK_MONOTONIC, &step_start);
            float loss = cross_entropy_loss(output, batch_labels, BATCH_SIZE);
            total_loss += loss;
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            stats.cross_entropy += get_time_diff(step_start, step_end);

            // Backward pass with timing
            backward_timed(nn, batch_input, hidden, output, batch_labels, BATCH_SIZE, &stats);
            
            // Weight update with timing
            update_weights_timed(nn, &stats);
        }
        
        printf("Epoch %d loss: %.4f\n", epoch, total_loss / num_batches);
    }
    
    // End total timing
    clock_gettime(CLOCK_MONOTONIC, &total_end);
    stats.total_time = get_time_diff(total_start, total_end);
    
    // Print detailed timing breakdown
    printf("\n=== C CPU IMPLEMENTATION TIMING BREAKDOWN ===\n");
    printf("Total training time: %.1f seconds\n\n", stats.total_time);
    
    printf("Detailed Breakdown:\n");
    printf("  Data loading:     %6.3fs (%5.1f%%)\n", stats.data_loading, 100.0 * stats.data_loading / stats.total_time);
    double forward_pass = stats.fwd_matmul1 + stats.fwd_bias1 + stats.fwd_relu + stats.fwd_matmul2 + stats.fwd_bias2 + stats.fwd_softmax;
    printf("  Forward pass:     %6.3fs (%5.1f%%)\n", forward_pass, 100.0 * forward_pass / stats.total_time);
    printf("  Loss computation: %6.3fs (%5.1f%%)\n", stats.cross_entropy, 100.0 * stats.cross_entropy / stats.total_time);
    double backward_pass = stats.bwd_output_grad + stats.bwd_matmul2 + stats.bwd_bias2 + stats.bwd_relu + stats.bwd_matmul1 + stats.bwd_bias1;
    printf("  Backward pass:    %6.3fs (%5.1f%%)\n", backward_pass, 100.0 * backward_pass / stats.total_time);
    printf("  Weight updates:   %6.3fs (%5.1f%%)\n", stats.weight_updates, 100.0 * stats.weight_updates / stats.total_time);
    
    free(hidden);
    free(output);
}

// Initialize weights using He initialization (random)
void initialize_random_weights(NeuralNetwork *nn) {
    initialize_weights(nn->weight1, INPUT_SIZE, HIDDEN_SIZE);
    initialize_weights(nn->weight2, HIDDEN_SIZE, OUTPUT_SIZE);
    initialize_bias(nn->bias1, HIDDEN_SIZE);
    initialize_bias(nn->bias2, OUTPUT_SIZE);
}

// Initialize neural network with random He weights
void initialize_neural_network(NeuralNetwork *nn) {
    nn->weight1 = malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    nn->weight2 = malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    nn->bias1 = malloc(HIDDEN_SIZE * sizeof(float));
    nn->bias2 = malloc(OUTPUT_SIZE * sizeof(float));
    nn->grad_weight1 = malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    nn->grad_weight2 = malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    nn->grad_bias1 = malloc(HIDDEN_SIZE * sizeof(float));
    nn->grad_bias2 = malloc(OUTPUT_SIZE * sizeof(float));

    initialize_random_weights(nn);
}

int main() {
    srand(time(NULL));  // Random seed for natural variance

    NeuralNetwork nn;
    initialize_neural_network(&nn);

    float *X_train = malloc(TRAIN_SIZE * INPUT_SIZE * sizeof(float));
    int *y_train = malloc(TRAIN_SIZE * sizeof(int));
    float *X_test = malloc(TEST_SIZE * INPUT_SIZE * sizeof(float));
    int *y_test = malloc(TEST_SIZE * sizeof(int));

    load_data("/home/soham/mnist_cuda_project/data/X_train.bin", X_train, TRAIN_SIZE * INPUT_SIZE);
    normalize_data(X_train, TRAIN_SIZE * INPUT_SIZE);
    load_labels("/home/soham/mnist_cuda_project/data/y_train.bin", y_train, TRAIN_SIZE);
    load_data("/home/soham/mnist_cuda_project/data/X_test.bin", X_test, TEST_SIZE * INPUT_SIZE);
    normalize_data(X_test, TEST_SIZE * INPUT_SIZE);
    load_labels("/home/soham/mnist_cuda_project/data/y_test.bin", y_test, TEST_SIZE);


    train_timed(&nn, X_train, y_train);

    free(nn.weight1);
    free(nn.weight2);
    free(nn.bias1);
    free(nn.bias2);
    free(nn.grad_weight1);
    free(nn.grad_weight2);
    free(nn.grad_bias1);
    free(nn.grad_bias2);
    free(X_train);
    free(y_train);
    free(X_test);
    free(y_test);

    return 0;
}