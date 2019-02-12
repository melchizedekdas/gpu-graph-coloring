#include<stdio.h>
#include<stdlib.h>
#include<float.h>
#include<math.h>
#include<cuda.h>
#include<curand_kernel.h>

#include<curand.h>
struct new_csr_graph{
	int v_count,*A, *IA, *color;
};
void init_input_graph(struct new_csr_graph *input_graph){
	input_graph->v_count=8;
	input_graph->A=(int *)malloc(sizeof(int)*16);
	input_graph->IA=(int *)malloc(sizeof(int)*9);
	input_graph->A[0]=1;
	input_graph->A[1]=2;
	input_graph->A[2]=0;
	input_graph->A[3]=2;
	input_graph->A[4]=0;
	input_graph->A[5]=1;
	input_graph->A[6]=3;
	input_graph->A[7]=2;
	input_graph->A[8]=4;
	input_graph->A[9]=3;
	input_graph->A[10]=5;
	input_graph->A[11]=6;
	input_graph->A[12]=7;
	input_graph->A[13]=4;
	input_graph->A[14]=4;
	input_graph->A[15]=4;

	input_graph->IA[0]=0;
	input_graph->IA[1]=2;
	input_graph->IA[2]=4;
	input_graph->IA[3]=7;
	input_graph->IA[4]=9;
	input_graph->IA[5]=13;
	input_graph->IA[6]=14;
	input_graph->IA[7]=15;
	input_graph->IA[8]=16;
}

__global__
void setup_random ( curandState * state, unsigned long seed , int v_count)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx<v_count){
		curand_init ( seed, idx, 0, &state[idx] );
	}
}

__global__
void generate_random( curandState* globalState, float * randomArray , int v_count)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx<v_count){
		curandState localState = globalState[idx];
		float RANDOM = curand_uniform( &localState );
		randomArray[idx] = RANDOM;
		globalState[idx] = localState;
	}
}

__global__
void init_kernel(int *d_IA, int *d_color, float *d_node_val, int v_count){
	int vertex_id=blockIdx.x*blockDim.x+threadIdx.x;
	if(vertex_id<v_count){
		d_node_val[vertex_id]=d_IA[vertex_id+1]-d_IA[vertex_id];
		d_color[vertex_id]=-1;
	}
}
__global__
void minmax_kernel(int *d_A, int *d_IA, int *d_color, float *d_node_val, float *d_min, float *d_max, int *d_cont, float max_val, int v_count){
	int vertex_id=blockIdx.x*blockDim.x+threadIdx.x;
	if(vertex_id<v_count){
		if (d_color[vertex_id] == -1) {
			float min = max_val, max = -1.0f;
			for (int edge_id = d_IA[vertex_id]; edge_id < d_IA[vertex_id + 1]; edge_id++) {
				int dest = d_A[edge_id];
				if (d_color[dest] == -1) {
					*d_cont = 1;
					min = min < d_node_val[dest] ? min : d_node_val[dest];
					max = max > d_node_val[dest] ? max : d_node_val[dest];
				}
			}
			d_min[vertex_id] = min;
			d_max[vertex_id] = max;
		}
	}
}
__global__
void color_kernel(int *d_color, float *d_node_val, float *d_min, float *d_max, int *d_change, int curr_color, int v_count){
	int vertex_id=blockIdx.x*blockDim.x+threadIdx.x;
	if(vertex_id<v_count && d_color[vertex_id]==-1){
		if(d_node_val[vertex_id]<d_min[vertex_id]){
			d_color[vertex_id]=curr_color;
			*d_change=1;
		}
		if(d_node_val[vertex_id]>d_max[vertex_id]){
			d_color[vertex_id]=1+curr_color;
			*d_change=1;
		}
	}
}
void assign_color(struct new_csr_graph *input_graph){
	int cont=1, change, color=0;
	int *d_A, *d_IA, *d_color, *d_cont, *d_change;
	float *d_node_val, *d_min, *d_max;
	cudaMalloc((void **)&d_A,input_graph->IA[input_graph->v_count]*sizeof(int));
	cudaMalloc((void **)&d_IA,(input_graph->v_count+1)*sizeof(int));
	cudaMalloc((void **)&d_color,input_graph->v_count*sizeof(int));
	cudaMalloc((void **)&d_cont,sizeof(int));
	cudaMalloc((void **)&d_change,sizeof(int));
	cudaMalloc((void **)&d_node_val,input_graph->v_count*sizeof(float));
	cudaMalloc((void **)&d_min,input_graph->v_count*sizeof(float));
	cudaMalloc((void **)&d_max,input_graph->v_count*sizeof(float));

	cudaMemcpy(d_A,input_graph->A,input_graph->IA[input_graph->v_count]*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(d_IA,input_graph->IA,(input_graph->v_count+1)*sizeof(int),cudaMemcpyHostToDevice);

	init_kernel<<<ceil(input_graph->v_count/256.0),256>>>(d_IA, d_color, d_node_val, input_graph->v_count);
	while(cont){
		change=0;
		cont=0;

		cudaMemcpy(d_cont,&cont,sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(d_change,&change,sizeof(int),cudaMemcpyHostToDevice);

		minmax_kernel<<<ceil(input_graph->v_count/256.0),256>>>(d_A, d_IA, d_color, d_node_val, d_min, d_max, d_cont, FLT_MAX, input_graph->v_count);
		color_kernel<<<ceil(input_graph->v_count/256.0),256>>>(d_color, d_node_val, d_min, d_max, d_change, color, input_graph->v_count);

		cudaMemcpy(&cont,d_cont,sizeof(int),cudaMemcpyDeviceToHost);
		cudaMemcpy(&change,d_change,sizeof(int),cudaMemcpyDeviceToHost);

		if(cont && !change){
			curandState* devStates;
			cudaMalloc((void **)&devStates, input_graph->v_count * sizeof(curandState));
			setup_random<<<ceil(input_graph->v_count/256.0),256>>>( devStates, time(NULL),  input_graph->v_count);
			generate_random<<<ceil(input_graph->v_count/256.0),256>>>( devStates, d_node_val,  input_graph->v_count);
			cudaFree(devStates);
		}
		else{
			color+=2;
		}
	}
	cudaMemcpy(input_graph->color,d_color,input_graph->v_count*sizeof(int),cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_IA);
	cudaFree(d_color);
	cudaFree(d_cont);
	cudaFree(d_change);
	cudaFree(d_node_val);
	cudaFree(d_min);
	cudaFree(d_max);
}
int main(){
	struct new_csr_graph input_graph;
	init_input_graph(&input_graph);
	assign_color(&input_graph);
	for(int i=0;i<input_graph.v_count;i++){
		printf("%d ",input_graph.color[i]);
	}
}
