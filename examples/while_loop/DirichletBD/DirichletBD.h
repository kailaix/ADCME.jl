void forward(double *uu, const int*dof, int d, const int*ii, const int*jj, const double*vv, int n){
    int flag1, flag2;
    for(int i=0;i<n;i++){
      flag1 = -1; flag2 = -1;
      for(int j=0;j<d;j++){
          if(ii[i]==dof[j]) flag1=ii[i];
          if(jj[i]==dof[j]) flag2=jj[i];
      } 
      if(flag1!=-1 && flag2!=-1 && (flag1==flag2)) uu[i] = 1.0;
      else if(flag1==-1 && flag2!=-1) uu[i] = 0.0;
      else if(flag1!=-1 && flag2==-1) uu[i] = 0.0;
      else uu[i] = vv[i];
    }
}

void backward(double *grad_vv, const double*grad_uu, const int*dof, int d, const int*ii, const int*jj, const double*vv, int n){
    int flag1, flag2;
    for(int i=0;i<n;i++){
      flag1 = -1; flag2 = -1;
      for(int j=0;j<d;j++){
          if(ii[i]==dof[j]) flag1=ii[i];
          if(jj[i]==dof[j]) flag2=jj[i];
      } 
      if(flag1!=-1 && flag2!=-1 && (flag1==flag2)) grad_vv[i] = 0.0;
      else if(flag1==-1 && flag2!=-1) grad_vv[i] = 0.0;
      else if(flag1!=-1 && flag2==-1) grad_vv[i] = 0.0;
      else grad_vv[i] = grad_uu[i];
    }
}

