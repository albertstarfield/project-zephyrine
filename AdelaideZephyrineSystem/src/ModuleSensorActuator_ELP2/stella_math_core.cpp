
extern "C" {
    // 0=Add, 1=Sub, 2=Mul, 3=Div
    void fast_calc(double a, int op, double b, double* res, int* err) {
        *err = 0;
        switch(op) {
            case 0: *res = a + b; break;
            case 1: *res = a - b; break;
            case 2: *res = a * b; break;
            case 3: 
                if (b == 0.0) *err = 1; 
                else *res = a / b; 
                break;
            default: *err = 2;
        }
    }
    int health() { return 777; }
}
