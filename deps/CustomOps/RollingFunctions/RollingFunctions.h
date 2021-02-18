
void RollingSumFunctionForward(
    double *out, const double *u, int window, int n){
    for(int i = 0; i < window; i++)
        out[0] += u[i];
    for (int i = window; i < n; i++){
        out[i - (window-1)] = out[i-window] + u[i] - u[i-window];
    }
}

void RollingSumFunctionBackward(
    double *grad_u, 
    const double *grad_out, 
    const double *out, const double *u, int window, int n){
    int N = n - window + 1;
    grad_u[0] = grad_out[0];
    for (int i = 1; i < n; i++){
        if (i < window){
            if (i < N)
                grad_u[i] = grad_u[i-1] + grad_out[i];
            else 
                grad_u[i] = grad_u[i-1];
        }else{
            if (i < N)
                grad_u[i] = grad_u[i-1] - grad_out[i-window] + grad_out[i];
            else {
                grad_u[i] = grad_u[i-1] - grad_out[i-window];
            }
        }    
    }
}

void RollingMeanFunctionForward(
    double *out, const double *u, int window, int n
){
    for(int i = 0; i < window; i++)
        out[0] += u[i];
    out[0] /= window;
    for (int i = window; i < n; i++){
        out[i - (window-1)] = (out[i-window] * window + u[i] - u[i-window])/window;
    }
}


void RollingMeanFunctionBackward(
    double *grad_u, 
    const double *grad_out, 
    const double *out, const double *u, int window, int n){
    RollingSumFunctionBackward(grad_u, grad_out, out, u, window, n);
    for(int i = 0; i < n; i++){
        grad_u[i] = grad_u[i] / window;
    }
}


double __var_window(const double *u, const double mean, int window){
    double out = 0.0;
    for(int i = 0; i < window; i++){
        out += (u[i] - mean)*(u[i]-mean);
    }
    return out / (window - 1);
}

void __var_window_backward(double * grad_u, double grad_out, const double *u, const double mean, int window){
    for(int i = 0; i < window; i++){
        for (int j = 0; j < window; j++){
            grad_u[j] += 2 * (mean-u[i]) *  1.0 / window * grad_out / (window-1);
        }
        grad_u[i] += 2 * (u[i] - mean) * grad_out / (window-1);
    }
}

void RollingVarFunctionBackward(
    double *grad_u, 
    const double *grad_out, 
    const double *out, const double *u, int window, int n
){
    auto mean = new double[n-window+1];
    RollingMeanFunctionForward(mean, u, window, n);
    for (int i = window-1; i < n; i++){
        __var_window_backward(grad_u+i-window+1, grad_out[i-window+1], u+i-window+1, mean[i-window+1], window);
    }
    delete [] mean;
}

void RollingVarFunctionForward(
    double *out, const double *u, int window, int n
){
    RollingMeanFunctionForward(out, u, window, n);
    for (int i = window-1; i < n; i++){
        out[i-window+1] = __var_window(u+i-window+1, out[i-window+1], window);
    }
}

void RollingStdFunctionForward(
    double *out, const double *u, int window, int n
){
    RollingVarFunctionForward(out, u, window, n);
    for(int i = 0; i < n - window+1; i++)
        out[i] = sqrt(out[i]);
}

void RollingStdFunctionBackward(
    double *grad_u, 
    const double *grad_out, 
    const double *out, const double *u, int window, int n
){
    auto mean = new double[n-window+1];
    RollingMeanFunctionForward(mean, u, window, n);
    for (int i = window-1; i < n; i++){
        __var_window_backward(grad_u+i-window+1, grad_out[i-window+1] * 1.0/2.0/out[i-window+1], u+i-window+1, mean[i-window+1], window);
    }
    delete [] mean;
}

void RollingFunctionForward(
    double *out, const double *u, int window, int n, string op
){
    if(op.compare("sum")==0){
        RollingSumFunctionForward(out, u, window, n);
    }else if(op.compare("mean")==0){
        RollingMeanFunctionForward(out, u, window, n);
    }else if(op.compare("var")==0){
        RollingVarFunctionForward(out, u, window, n);
    }else if(op.compare("std")==0){
        RollingStdFunctionForward(out, u, window, n);
    }else{
        throw "invalid op in rolling function";
    }
}

void RollingFunctionBackward(
    double *grad_u, 
    const double *grad_out, 
    const double *out, const double *u, int window, int n, string op
){
    if(op.compare("sum")==0){
        RollingSumFunctionBackward(grad_u, grad_out, out, u, window, n);
    }else if(op.compare("mean")==0){
        RollingMeanFunctionBackward(grad_u, grad_out, out, u, window, n);
    }else if(op.compare("var")==0){
        RollingVarFunctionBackward(grad_u, grad_out, out, u, window, n);
    }else if(op.compare("std")==0){
        RollingStdFunctionBackward(grad_u, grad_out, out, u, window, n);
    }else{
        throw "invalid op in rolling function";
    }
}