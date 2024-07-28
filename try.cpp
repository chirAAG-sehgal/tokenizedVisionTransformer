// import numpy as np

// # Function to sample from a Gaussian distribution centered at a given index
// def sample_from_gaussian(index, sigma=2, num_samples=1):
//     # Generate samples from the Gaussian distribution
//     samples = np.random.normal(loc=index, scale=sigma, size=num_samples)
    
//     # Round and cast to integer
//     sampled_indexes = np.round(samples).astype(int)
    
//     return sampled_indexes

// # Example usage
// index = 23
// sigma = 2  # Standard deviation of the Gaussian
// num_samples = 10  # Number of samples to generate

// sampled_indexes = sample_from_gaussian(index, sigma, num_samples)
// print(f"Original index: {index}")
// print(f"Sampled indexes: {sampled_indexes}")

#include <iostream>
#include <vector>
#include <algorithm>
#include <bits/stdc++.h>

using namespace std;

// void minColoumns(int n, int k){
//     int count = k;
//     int actual_count = 0;
//     for(int i = n; i>0; i--){
//         if (i ==n){
//             count = count-i;
//             actual_count++;
//         }
//         else{
//             count = count-2*i;
//             actual_count+=2;
//         }
//         if (count<=0){
//             if (count == 0){
//                 break;
//             }
//             else{
//                 if (count <= (-i+1)){
//                     actual_count--;
//                     // cout<<actual_count<<" where it should be";
//                     break;
//                 }
//                 else{
//                     break;
//                 }
//             }
//         }
//     }
//     cout<<actual_count<<endl;
// }

int main(){
    // int n;
    // vector<int> arr(n) ;
    // int i, j, curr;
    // i = 0, j =1;
    // curr = arr[i];
    // sort(arr.begin(), arr.end(), greater<int>());
    // int flag = 10;
    // while (flag !=0){
    //     if(arr[j] - arr[i] >1){
            
    //     }
    // }

    return 0;
}