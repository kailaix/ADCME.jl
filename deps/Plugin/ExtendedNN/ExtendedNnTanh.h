void forward_tanh(
    double *out, double *sensitivity,
    const double *x, int n, 
    const int64* config, int m, const double *theta){
    const std::lock_guard<std::mutex> lock(mu);
   Stack stack;
   Array<2, double, true> X(n, config[0]);
   std::vector<Array<2, double, true>*> W(m-1);
   std::vector<Array<1, double, true>*> b(m-1);
   int k = 0;
   for(int i=0;i<m-1;i++){
     W[i] = new Array<2,double,true>(config[i], config[i+1]);
     b[i] = new Array<1, double,true>(config[i+1]);
     for(int p=0;p<config[i];p++)
        for(int q=0;q<config[i+1];q++)
          (*W[i])(p,q) = theta[k++];
     for(int p=0;p<config[i+1];p++) (*b[i])(p) = theta[k++];
   }
   k = 0;
   for(int i=0;i<n;i++)
    for(int j=0;j<config[0];j++)
       X(i,j) = x[k++];

   stack.new_recording();
   std::vector<Array<2, double, true>*> y1(m-1);
   std::vector<Array<2, double, true>*> y2(m-1);
   
   for(int i=0;i<m-1;i++){
     y1[i] = new Array<2,double, true>(n, config[i+1]);
     y2[i] = new Array<2,double, true>(n, config[i+1]);
     if (i==0) 
        *(y1[i]) = X**(*W[i]);
     else 
        *(y1[i]) = (*y2[i-1])**(*W[i]);
     for(int p=0;p<n;p++){
       for(int q=0;q<config[i+1];q++)
        if (i==m-2)
          (*y2[i])(p,q) = ((*y1[i])(p,q) + (*b[i])(q));
        else
          (*y2[i])(p,q) = tanh((*y1[i])(p,q) + (*b[i])(q));
     }
   }

   k = 0;
   for(int i=0;i<n;i++){
     for(int j=0;j<config[m-1];j++)
      out[k++] = (*y2[m-2])(i,j).value();
   }

  auto temp = sum(*y2[m-2], 0);
  int k1 = 0;
  //  compute sensitivity
  for (int s=0;s<config[m-1];s++){
    stack.clear_gradients();
    for(int i=0;i<config[m-1];i++){
      if(i==s) temp(i).set_gradient(1.0);
      else temp(i).set_gradient(0.0);
    }
    stack.compute_adjoint();

    auto grad_X = X.get_gradient();
    for(int p=0;p<n;p++)
      for(int q=0;q<config[0];q++){
          int idx = p*config[0]*config[m-1] + s*config[0] + q;
          sensitivity[idx] = grad_X(p,q);
      }
        // sensitivity[k1++] = grad_X(p, q);
  }

   for(int i=0;i<m-1;i++){
     delete W[i];
     delete b[i];
     delete y1[i];
     delete y2[i];
   }

}


void backward_tanh(
    double *grad_x, 
    double *grad_theta,
    const double *grad_out, 
    const double *out, const double *sensitivity,
    const double *x, int n, 
    const int64* config, int m, const double *theta, int n_theta){
  const std::lock_guard<std::mutex> lock(mu);
  /*==========================================================*/
  Stack stack;
  Array<2, double, true> X(n, config[0]);
  std::vector<Array<2, double, true>*> W(m-1);
  std::vector<Array<1, double, true>*> b(m-1);
  int k = 0;
  for(int i=0;i<m-1;i++){
    W[i] = new Array<2,double,true>(config[i], config[i+1]);
    b[i] = new Array<1, double,true>(config[i+1]);
    for(int p=0;p<config[i];p++)
      for(int q=0;q<config[i+1];q++)
        (*W[i])(p,q) = theta[k++];
    for(int p=0;p<config[i+1];p++) (*b[i])(p) = theta[k++];
  }
  k = 0;
  for(int i=0;i<n;i++)
  for(int j=0;j<config[0];j++)
      X(i,j) = x[k++];

  stack.new_recording();
  std::vector<Array<2, double, true>*> y1(m-1);
  std::vector<Array<2, double, true>*> y2(m-1);
  
  for(int i=0;i<m-1;i++){
    y1[i] = new Array<2,double, true>(n, config[i+1]);
    y2[i] = new Array<2,double, true>(n, config[i+1]);
    if (i==0) 
      *(y1[i]) = X**(*W[i]);
    else 
      *(y1[i]) = (*y2[i-1])**(*W[i]);
    for(int p=0;p<n;p++){
      for(int q=0;q<config[i+1];q++)
      if (i==m-2)
        (*y2[i])(p,q) = ((*y1[i])(p,q) + (*b[i])(q));
      else
        (*y2[i])(p,q) = tanh((*y1[i])(p,q) + (*b[i])(q));
    }
  }
  /*==========================================================*/

  auto res = *y2[m-2];
  adouble l = 0.0;
  k = 0;
  for(int p=0;p<n;p++){
    for(int q=0;q<config[m-1];q++)
      l += res(p,q)*grad_out[k++];
  }
  l.set_gradient(1.0);
  stack.compute_adjoint();
  k = 0;
  for(int i=0;i<m-1;i++){
    auto grad_W = (*W[i]).get_gradient();
    for(int p=0;p<config[i];p++)
      for(int q=0;q<config[i+1];q++)
        grad_theta[k++] = grad_W(p,q);
    auto grad_b = (*b[i]).get_gradient();
    for(int p=0;p<config[i+1];p++)
      grad_theta[k++] = grad_b(p);
  }

  auto grad_X = X.get_gradient();
  k = 0;
  for(int i=0;i<n;i++)
    for(int j=0;j<config[0];j++){
      grad_x[k++] = grad_X(i, j);
    }

  for(int i=0;i<m-1;i++){
    delete W[i];
    delete b[i];
    delete y1[i];
    delete y2[i];
  }

}
