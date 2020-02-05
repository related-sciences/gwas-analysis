//import $ivy.`io.projectglow::glow:0.2.0`

// Use local Glow build with fix for index overflow in large PLINK datasets; 
// this will likely be upstream in the Glow 0.2.1 release
// See: https://github.com/projectglow/glow/issues/133
import $ivy.`io.projectglow::glow:0.2.1-SNAPSHOT`