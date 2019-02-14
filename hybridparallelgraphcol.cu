#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda.h>
#include<curand_kernel.h>

#define NO_COLOR 0
#define MIN_COLOR -1
#define MAX_COLOR 1

struct new_csr_graph{
	int v_count,*A, *IA, *color;
};

__global__
void init_kernel(int *d_color, float *d_node_val, int *d_IA, int v_count){
	int vertex_id=blockIdx.x*blockDim.x+threadIdx.x;
	if(vertex_id<v_count){
		d_node_val[vertex_id]=d_IA[vertex_id+1]-d_IA[vertex_id];
		d_color[vertex_id]=NO_COLOR;
	}
}
__global__
void random_generate(float *d_node_val, curandState* state, unsigned long seed, int v_count){
	int vertex_id=blockIdx.x*blockDim.x+threadIdx.x;
	if(vertex_id<v_count){
		curand_init ( seed, vertex_id, 0, &state[vertex_id] );
		d_node_val[vertex_id]=curand_uniform(state+vertex_id);
	}
}
__global__
void minmax_kernel(int *d_A, int *d_IA, int *d_color, float *d_node_val, char *d_color_code, char *d_cont, char *d_change, int v_count){
	int vertex_id=blockIdx.x*blockDim.x+threadIdx.x;
	if(vertex_id<v_count && d_color[vertex_id]==NO_COLOR){
		int total=d_IA[vertex_id+1];
		float curr_node_val=d_node_val[vertex_id];
		float edge_node_val;
		char is_min=1, is_max=1;
		for(int i=d_IA[vertex_id];i<total;i++){
			edge_node_val=d_node_val[d_A[i]];
			if(edge_node_val<=curr_node_val){
				is_min=0;
			}
			if(edge_node_val>=curr_node_val){
				is_max=0;
			}
		}
		if(is_min){
			d_color_code[vertex_id]=MIN_COLOR;
			*d_change=1;
		}
		else if(is_max){
			d_color_code[vertex_id]=MAX_COLOR;
			*d_change=1;
		}
		else{
			d_color_code[vertex_id]=NO_COLOR;
			*d_cont=1;
		}
	}
}
__global__
void color_kernel(int *d_color, char *d_color_code, int curr_color, int v_count){
	int vertex_id=blockIdx.x*blockDim.x+threadIdx.x;
	if(vertex_id<v_count && d_color[vertex_id]==NO_COLOR){
		if(d_color_code[vertex_id]==MIN_COLOR){
			d_color[vertex_id]=curr_color;
		}
		else if(d_color_code[vertex_id]==MAX_COLOR){
			d_color[vertex_id]=curr_color+1;
		}
	}
}
void assign_color(struct new_csr_graph *input_graph){
	int cur_color=NO_COLOR+1;
	char cont=1, change;
	int *d_A, *d_IA, *d_color;
	char *d_cont, *d_change, *d_color_code;
	float *d_node_val;
	cudaMalloc((void **)&d_A,input_graph->IA[input_graph->v_count]*sizeof(int));
	cudaMalloc((void **)&d_IA,(input_graph->v_count+1)*sizeof(int));
	cudaMalloc((void **)&d_color,input_graph->v_count*sizeof(int));
	cudaMalloc((void **)&d_cont,sizeof(char));
	cudaMalloc((void **)&d_change,sizeof(char));
	cudaMalloc((void **)&d_color_code,input_graph->v_count*sizeof(char));
	cudaMalloc((void **)&d_node_val,input_graph->v_count*sizeof(float));

	cudaMemcpy(d_A,input_graph->A,input_graph->IA[input_graph->v_count]*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(d_IA,input_graph->IA,(input_graph->v_count+1)*sizeof(int),cudaMemcpyHostToDevice);

	init_kernel<<<ceil(input_graph->v_count/256.0),256>>>(d_color, d_node_val, d_IA, input_graph->v_count);

	while(cont){
		cont=0;
		change=0;
		cudaMemcpy(d_cont,&cont,sizeof(char),cudaMemcpyHostToDevice);
		cudaMemcpy(d_change,&change,sizeof(char),cudaMemcpyHostToDevice);
		minmax_kernel<<<ceil(input_graph->v_count/256.0),256>>>(d_A, d_IA, d_color, d_node_val, d_color_code, d_cont, d_change, input_graph->v_count);
		color_kernel<<<ceil(input_graph->v_count/256.0),256>>>(d_color, d_color_code, cur_color, input_graph->v_count);
		cudaMemcpy(&cont,d_cont,sizeof(char),cudaMemcpyDeviceToHost);
		cudaMemcpy(&change,d_change,sizeof(char),cudaMemcpyDeviceToHost);
		if(cont && !change){
			curandState* d_states;
			cudaMalloc((void **)&d_states, input_graph->v_count * sizeof(curandState));
			random_generate<<<ceil(input_graph->v_count/256.0),256>>>(d_node_val, d_states, time(NULL), input_graph->v_count);
			cudaFree(d_states);
		}
		else{
			cur_color+=2;
		}
	}
	cudaFree(d_A);
	cudaFree(d_IA);
	cudaFree(d_cont);
	cudaFree(d_node_val);
	cudaMemcpy(input_graph->color,d_color,input_graph->v_count*sizeof(int),cudaMemcpyDeviceToHost);
	cudaFree(d_color);
}

void init_input_graph(struct new_csr_graph *input_graph){
	input_graph->v_count=8;
	input_graph->A=(int *)malloc(sizeof(int)*16);
	input_graph->IA=(int *)malloc(sizeof(int)*9);
	input_graph->color=(int *)malloc(sizeof(int)*input_graph->v_count);
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
int main(){
	struct new_csr_graph input_graph;
	init_input_graph(&input_graph);
	assign_color(&input_graph);
	printf("\nresult coloring:");
	for(int i=0;i<input_graph.v_count;i++){
		printf("%d ",input_graph.color[i]);
	}
}
