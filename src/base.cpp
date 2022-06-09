#include <iostream>

#include <Eigen/Dense>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include <CGAL/Cartesian_d.h>
#include <CGAL/hilbert_sort.h>
#include <CGAL/Spatial_sort_traits_adapter_d.h>
#include <vector>
#include <CGAL/boost/iterator/counting_iterator.hpp>


typedef CGAL::Cartesian_d<double>                                        Kernel;
typedef Kernel::Point_d                                                 Point_d;
typedef CGAL::Spatial_sort_traits_adapter_d<Kernel, Point_d*>   Search_traits_d;

using namespace Eigen;



//obtain order based on Hilbert Curve Projection
void hilbert_curve_order_median(const double * X, int n, int d, int * orderX)
{

    std::vector<Point_d> points;
    double * temp = new double[d];
    int i = 0,j = 0;

    for (i = 0; i < n; i++ ) {
      for (j = 0; j < d; j++) {
        temp[j] = X[d * i + j];
      }
      points.push_back(Point_d(d, temp, temp+d));
    }

    std::vector<std::ptrdiff_t> indices;
    indices.reserve(points.size());

    std::copy(
      boost::counting_iterator<std::ptrdiff_t>(0),
      boost::counting_iterator<std::ptrdiff_t>(points.size()),
      std::back_inserter(indices) );

    // If you want to use 'middle policy', please refer to https://github.com/CGAL/cgal/discussions/6464 and then choose CGAL::Hilbert_sort_middle_policy().
    CGAL::hilbert_sort (indices.begin(),
                        indices.end(), 
                        Search_traits_d( &(points[0]) ),
                        CGAL::Hilbert_sort_median_policy());

    for (i = 0; i < n; i++) {
      orderX[i] = indices[i];
    }

    delete [] temp;
    temp=NULL;
}
 
ArrayXi Hilbert_Curve_Order(const MatrixXd & X1)
{
    int d1 = X1.rows();
    int n1 = X1.cols();
 
    ArrayXi orderX1 = ArrayXi::Zero(n1);   
    hilbert_curve_order_median(X1.data(), n1, d1, &orderX1[0]);

    return orderX1;
}



//North-West Corner Algorithm
MatrixXd general_plan(const double * a_weight, int n, const double * b_weight, int m){
    
    int i = 0, j = 0, l = n+m-1, c_id = 0;
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
    MatrixXd G(3, c_id);

    for(int s=0; s<c_id; s++){
      G(0,s) = idx1[s];
      G(1,s) = idx2[s];
      G(2,s) = idx3[s];
    }

    return G;
}

MatrixXd General_Plan(const VectorXd & X, const VectorXd & Y)
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
    // m.def("hilbert_sort_middle", &Hilbert_Sort_Middle_CGAL);
    m.def("general_Plan", &General_Plan);
}