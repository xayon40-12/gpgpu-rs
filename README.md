# gpgpu-rs  
A simple GPU computing library based on OpenCL.  

Can be used more as a test than for real purpose yet.  
Examples can be executed witch `cargo run --example [name]`. For instance:  
`cargo run --example random`  

# Features  
- [ ] comments and documentations  

Global:  
- [x] kernels  
- [x] algorithms  
- [x] functions  
- [x] file loading (CPU)  
- [x] file loading (GPU) max elements: 8192  
- [x] file loading interpolation (high extrema excluded)  
- [x] huge file loading (GPU)  
- [x] Serialize/Deserialize for kernels and functions  
- [x] Integrators  

Kernels:  
- [x] component wise operators buffer/buffer  
- [x] component wise operators buffer/constant  
- [ ] component wise operators for each buffer types (currently only F64)  
- [x] correlation function  

Algorithms:  
- [x] sum  
- [x] min/max  
- [x] moments  
- [x] cumulants  
- [x] chose on which dimension to apply algorithms  
- [x] 1,2,3D Fourier transform  

Integrators:  
- [ ] PDE generator
- [ ] Euler  
- [ ] QSS  

Random number generation:  
- [x] philox 4x32,2x64,4x64  
- [x] uniform and normal distribution  
