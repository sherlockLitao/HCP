#include <iostream>
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <algorithm>


// split
class comp
{
    friend class Hilbert_sort;
    public:

      int current_axis1;
      bool sign1;
      const Eigen::ArrayXXd & X1;


    public:

      comp (int current_axis11, bool sign11, const Eigen::ArrayXXd & X11): current_axis1(current_axis11), sign1(sign11), X1(X11){}
      
      bool operator()(const int &i1, const int &i2)
      {
        return (sign1 ? ( X1(i1,current_axis1) >  X1(i2,current_axis1) )
                      : ( X1(i1,current_axis1) <  X1(i2,current_axis1) ));
      }


};


//using recursive sort algorithm from CGAL library
class Hilbert_sort
{
  public:

    const Eigen::ArrayXXd & X;
    int d;
    int n;
    ptrdiff_t pow_d;


  public:

    Hilbert_sort (const Eigen::ArrayXXd & X1):X(X1){}

    void hilbert_median_sort_d(std::vector<ptrdiff_t>::iterator begin, std::vector<ptrdiff_t>::iterator end, int first_axis, std::vector<bool> signs){
        
        if (end - begin <= 1) return;

        int i = 0, ii = 0, j = 0, jj = 0;
        bool sign = false;
        int _d = d;
        ptrdiff_t _pow_d = pow_d;

        if ((end - begin) < (_pow_d/2)) { 
          _pow_d = 1;
          _d = 0;
          while ( (end-begin) > _pow_d) {
            _d++;
            _pow_d *= 2;
          }
        }

        
        // split at 1+2+4+...+2^{_d-1} index and assign the first axis
        std::vector<std::vector<ptrdiff_t>::iterator> split_index(_pow_d+1);
        split_index[0] = begin;
        split_index[_pow_d] = end;

        std::vector<int> axiss(_pow_d+1);
        int current_axis = first_axis;
        int current_step = _pow_d;
        int last_step = 0;

        for(i=0; i<_d; i++){
          last_step = current_step;
          current_step = current_step/2;
          sign = signs[current_axis]; 
          for(j=0; j<pow(2,i); j++){
            jj = current_step + last_step * j;
            axiss[jj] = current_axis;
            if(split_index[jj-current_step] >= split_index[jj+current_step])split_index[jj] = split_index[jj - current_step];
            else{                        
              std::vector<ptrdiff_t>::iterator _med = split_index[jj-current_step] + (split_index[jj+current_step] - split_index[jj-current_step]) / 2;
              comp cmp(current_axis, sign, X);
              std::nth_element (split_index[jj-current_step], _med, split_index[jj+current_step], cmp);
              split_index[jj] = _med;
            }
            sign = !sign;
          }
          current_axis = (current_axis+1)%d;
        }


        if((end-begin)<pow_d) return;

        // perform recursive sort
        int last_axis = (first_axis+d-1)%d;
        hilbert_median_sort_d(split_index[0], split_index[1], last_axis, signs);

        for(i=1; i<pow_d-1; i=i+2){
          ii = axiss[i+1];
          hilbert_median_sort_d(split_index[i], split_index[i+1], ii, signs);
          hilbert_median_sort_d(split_index[i+1], split_index[i+2], ii, signs);
          signs[ii] = !signs[ii];
          signs[last_axis] = !signs[last_axis];
        }

       hilbert_median_sort_d(split_index[pow_d-1], split_index[pow_d], last_axis, signs);

    }

    void Hilbert_sort_median_d(std::vector<ptrdiff_t>::iterator begin, std::vector<ptrdiff_t>::iterator end)
    {
      n = X.rows() * 2;
      d = X.cols();
      pow_d = 1;
      int i=0;
      std::vector<bool> direction(d,false);

      for (i=0; i<d; i++) {
        pow_d *= 2;        
        n/=2;
        if(n==0)
          break;
      }

      hilbert_median_sort_d (begin, end, 0, direction);
    }

};



//Hilbert order
Eigen::ArrayXi Hilbert_Curve_Order(const Eigen::ArrayXXd & X)
{
    
    std::vector<ptrdiff_t> indices(X.rows());
    int i = 0;

    for(i=0; i<X.rows(); i++){
      indices[i] = i;
    }

    Hilbert_sort Hs(X);
    Hs.Hilbert_sort_median_d(indices.begin(),indices.end());
    
    Eigen::ArrayXi res = Eigen::ArrayXi::Zero(X.rows());
    for (int i = 0; i < X.rows(); i++) {
      res(i) = indices[i];
    }

    return res;
}




//North-West Corner Algorithm
Eigen::ArrayXXd general_plan(const double * a_weight, int n, const double * b_weight, int m){
    
    int i = 0, j = 0, l = n+m-1, c_id = 0, s = 0;
    double w_i = a_weight[0], w_j = b_weight[0];

    std::vector<double> idx1(l);
    std::vector<int> idx2(l);
    std::vector<int> idx3(l);

    while(true){

        if(w_i < w_j || j == m - 1){
        
            idx1[c_id] = w_i;
            idx2[c_id] = i;
            idx3[c_id] = j;
            i++;
            if(i == n){break;}
            w_j -= w_i;
            w_i = a_weight[i];
        }
        else{
            idx1[c_id] = w_j;
            idx2[c_id] = i;
            idx3[c_id] = j;
            j++;
            if(j == m){break;}
            w_i -= w_j;
            w_j = b_weight[j];
        }
        c_id++;
            
    }

    c_id++;
    Eigen::ArrayXXd G(3, c_id);

    for(s=0; s<c_id; s++){
      G(0,s) = idx1[s];
      G(1,s) = idx2[s];
      G(2,s) = idx3[s];
    }

    return G;
}

Eigen::ArrayXXd General_Plan(const Eigen::ArrayXd & X, const Eigen::ArrayXd & Y)
{
    int n1 = X.rows();
    int m1 = Y.rows();

    return  general_plan(X.data(), n1, Y.data(), m1);
}





namespace py = pybind11;
PYBIND11_MODULE(base, m)
{
    m.doc() = "source code";
    m.def("hilbert_order", &Hilbert_Curve_Order);
    m.def("general_Plan", &General_Plan);
}
