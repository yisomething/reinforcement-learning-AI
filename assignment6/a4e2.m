close all;
clear all;

r=2;
p0=0.1;

for i=1:20
    
    d=Jk(p0,r)\(-error(p0,r));
    r=r+d(1)
    p0=p0+d(2)
    if norm(error(p0,r))<0.0001
        disp('converge');
    end
    
end


function F=error(p0,r)

F=[0.19-p0*(r.^1);
    0.36-p0*(r.^2);
    0.69-p0*(r.^3);
    1.3-p0*(r.^4);
    2.5-p0*(r.^5);
    4.7-p0*(r.^6);
    8.5-p0*(r.^7);
    14-p0*(r.^8);];

end

function J=Jk(p0,r)

J=[-p0*1*(r^(1-1)) -r^1 ;
   -p0*2*(r^(2-1)) -r^2 ;
   -p0*3*(r^(3-1)) -r^3 ;
   -p0*4*(r^(4-1)) -r^4 ;
   -p0*5*(r^(5-1)) -r^5 ;
   -p0*6*(r^(6-1)) -r^6 ;
   -p0*7*(r^(7-1)) -r^7 ;
   -p0*8*(r^(8-1)) -r^8 ;];

end
