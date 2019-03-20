#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda.h>
#include<curand_kernel.h>
#include <time.h>

#define NO_COLOR 0
#define MIN_COLOR -1
#define MAX_COLOR 1

struct new_csr_graph{
	int v_count,*A, *IA, *color;
};

__global__
void init_kernel(int *d_color, float *d_node_val, curandState* state, unsigned long seed, int v_count){
	int vertex_id=blockIdx.x*blockDim.x+threadIdx.x;
	if(vertex_id<v_count){
		curand_init ( seed, vertex_id, 0, &state[vertex_id] );
		d_node_val[vertex_id]=curand_uniform(state+vertex_id);
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
			if(d_color[d_A[i]]!=NO_COLOR){
				//if this adjacent vertex is already colored then continue
				continue;
			}
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


	curandState* d_states;
	cudaMalloc((void **)&d_states, input_graph->v_count * sizeof(curandState));
	init_kernel<<<ceil(input_graph->v_count/256.0),256>>>(d_color, d_node_val, d_states, time(NULL), input_graph->v_count);
	cudaFree(d_states);

	int rand_ver=1;
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
			cudaMalloc((void **)&d_states, input_graph->v_count * sizeof(curandState));
			random_generate<<<ceil(input_graph->v_count/256.0),256>>>(d_node_val, d_states, time(NULL)+rand_ver++, input_graph->v_count);
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
int init_input_graph(struct new_csr_graph *input_graph, char *file_name){
	struct edge{
		int vertex1, vertex2;
	}*edge_list;
	FILE *file_pointer ;
	// in read mode using "r" attribute
	file_pointer = fopen(file_name, "r") ;
	if ( file_pointer == NULL )
	{
		return 1;
	}
	char new_line_flag=1, line_type=0, c;
	int param1=0, param2=0;
	int phase=1, edge_id=0, e_count, i;
	for (c = getc(file_pointer); c != EOF; c = getc(file_pointer))
	{
		if(c=='\n'){
			new_line_flag=1;
			if(line_type=='p'){
				input_graph->v_count=param1;
				e_count=param2;
				input_graph->IA=(int *)malloc((param1+1)*sizeof(int));
				input_graph->A=(int *)malloc(param2*2*sizeof(int));
				input_graph->color=(int *)malloc(param1*sizeof(int));
				for(i=0;i<=param1;i++){
					input_graph->IA[i]=0;
				}
				edge_list=(struct edge *)malloc(param2*sizeof(struct edge));
			}
			else if(line_type=='e'){
				edge_list[edge_id].vertex1=param1-1;
				edge_list[edge_id].vertex2=param2-1;
				input_graph->IA[param1]++;
				input_graph->IA[param2]++;
				edge_id++;
			}

			param1=0, param2=0;
			line_type=0;
			continue;
		}
		if(new_line_flag){
			line_type=c;
			phase=1;
			new_line_flag=0;
			continue;
		}
		if(line_type=='e'){
			switch(phase){
			case 1:
				if(c>='0' && c<='9'){
					param1=c-'0';
					phase++;
				}
				break;
			case 2:
				if(c>='0' && c<='9'){
					param1=param1*10+c-'0';
				}
				else{
					phase++;
				}
				break;
			case 3:
				if(c>='0' && c<='9'){
					param2=param2*10+c-'0';
				}
				else{
					phase++;
				}
				break;
			}
		}

		else if(line_type=='p'){
			switch(phase){
			case 1:
				if(c>='0' && c<='9'){
					param1=c-'0';
					phase++;
				}
				break;
			case 2:
				if(c>='0' && c<='9'){
					param1=param1*10+c-'0';
				}
				else{
					phase++;
				}
				break;
			case 3:
				if(c>='0' && c<='9'){
					param2=param2*10+c-'0';
				}
				else{
					phase++;
				}
				break;
			}
		}
	}
	fclose(file_pointer) ;
	if(!new_line_flag && line_type=='e'){
		edge_list[edge_id].vertex1=param1-1;
		edge_list[edge_id].vertex2=param2-1;
		input_graph->IA[param1]++;
		input_graph->IA[param2]++;
	}
	int *vertex_p=(int *)malloc(input_graph->v_count*sizeof(int));
	for(i=0;i<input_graph->v_count;i++){
		input_graph->IA[i+1]+=input_graph->IA[i];
		vertex_p[i]=0;
	}
	for(edge_id=0;edge_id<e_count;edge_id++){
		input_graph->A[input_graph->IA[edge_list[edge_id].vertex1]+(vertex_p[edge_list[edge_id].vertex1]++)]=edge_list[edge_id].vertex2;
		input_graph->A[input_graph->IA[edge_list[edge_id].vertex2]+(vertex_p[edge_list[edge_id].vertex2]++)]=edge_list[edge_id].vertex1;
	}
	free(edge_list);
	free(vertex_p);
	return 0;
}
int validate_coloring(struct new_csr_graph *input_graph){
	for(int i=0;i<input_graph->v_count;i++){
		for(int j=input_graph->IA[i];j<input_graph->IA[i+1];j++){
			if(input_graph->color[i]==input_graph->color[input_graph->A[j]]){
				return 0;
			}
		}
	}
	return 1;
}
int count_colors(struct new_csr_graph *input_graph){
	int max_color_used=0;
	for(int i=0;i<input_graph->v_count;i++){
		max_color_used=max_color_used>input_graph->color[i]?max_color_used:input_graph->color[i];
	}
	char *color_used=(char *)malloc(sizeof(char)*max_color_used);
	int total_colors=0;
	for(int i=0;i<max_color_used;i++){
		color_used[i]=0;
	}
	for(int i=0;i<input_graph->v_count;i++){
		color_used[input_graph->color[i]]=1;
	}
	for(int i=0;i<max_color_used;i++){
		if(color_used[i]==1){
			total_colors++;
		}
	}
	return total_colors;
}
int main(){
	struct new_csr_graph input_graph;
	init_input_graph(&input_graph, "input.txt");

	clock_t start, end;
	double cpu_time_used;
	start = clock();

	assign_color(&input_graph);

	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

	printf("\ntime taken:%f",cpu_time_used);
	if(!validate_coloring(&input_graph)){
		printf("\nInvalid coloring!");
		return 0;
	}
	printf("\nNo. of colors used:%d",count_colors(&input_graph));
	printf("\nresult coloring:");
	for(int i=0;i<input_graph.v_count;i++){
		printf("%d ",input_graph.color[i]);
	}
}
