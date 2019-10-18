void forward(double *uu, const int64*dof, int64 d, const int64*ii, const int64*jj, const double*vv, int64 n){
  // assumption: d and ii increase
    bool flag = false;
    int j = 0;
    int i = 0;
    for(;;){
      
      if(i==n) return;

      if(j==d){
        uu[i] = vv[i];
        i++;
        continue;
      }

      if(ii[i]>dof[j]){
        j++;
        flag = false;
        continue;
      }

      if(ii[i]<dof[j]){
        uu[i] = vv[i];
        i++;
        continue;
      }

      if(jj[i]!=dof[j]){
        uu[i] = 0.0;
        i++;
        continue;
      }

      if(jj[i]==dof[j]){
        if (!flag){
          uu[i] = 1.0;
          flag = true;
          i++;
        }
        else{
          uu[i] = 0.0;
          i++;
        }
        continue;
      }

      printf("Shouldn't be here");
      return;

    }
    
}

void backward(double *grad_vv, const double*grad_uu, const int64*dof, int64 d, const int64*ii, const int64*jj, const double*vv, int64 n){
    bool flag = false;
    int j = 0;
    int i = 0;
    for(;;){
      
      if(i==n) return;

      if(j==d){
        grad_vv[i] = grad_uu[i];
        i++;
        continue;
      }

      if(ii[i]>dof[j]){
        j++;
        flag = false;
        continue;
      }

      if(ii[i]<dof[j]){
        grad_vv[i] = grad_uu[i];
        i++;
        continue;
      }

      if(jj[i]!=dof[j]){
        grad_vv[i] = 0.0;
        i++;
        continue;
      }

      if(jj[i]==dof[j]){
        if (!flag){
          grad_vv[i] = 0.0;
          flag = true;
          i++;
        }
        else{
          grad_vv[i] = 0.0;
          i++;
        }
        continue;
      }

      printf("Shouldn't be here");
      return;

    }
}

