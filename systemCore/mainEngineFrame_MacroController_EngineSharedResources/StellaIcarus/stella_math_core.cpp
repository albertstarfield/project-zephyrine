
extern "C" {
    // Error Codes: 0=Success, 1=DivByZero, 2=UnknownOp
    void perform_calc(double n1, int op_code, double n2, double* result, int* error_code) {
        *error_code = 0;
        *result = 0.0;
        
        switch(op_code) {
            case 0: // Add
                *result = n1 + n2;
                break;
            case 1: // Sub
                *result = n1 - n2;
                break;
            case 2: // Mul
                *result = n1 * n2;
                break;
            case 3: // Div
                if (n2 == 0.0) {
                    *error_code = 1;
                } else {
                    *result = n1 / n2;
                }
                break;
            default:
                *error_code = 2;
        }
    }
    
    // Simple health check to ensure ABI compatibility
    int health_check() { return 1337; }
}
