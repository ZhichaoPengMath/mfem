function GenerateCartisenMesh(x_left,x_right,y_bottom,y_top, mesh_number)

%this is a simple matlab scrip to generate 2^n meshes
% partition in one direction
m= 3*2^mesh_number;

% total elements number
elements=m^2;




%boundary 2*m

dx = (x_right - x_left )/m;
dy = (y_top - y_bottom)/m;

xx=x_left:dx:x_right;
yy=y_bottom:dy:y_top;

fileID = fopen('test.mesh','w');

fprintf(fileID, 'MFEM mesh v1.0\n'); % format
fprintf(fileID, '\ndimension\n2\n'); % dimensions
fprintf(fileID, '\nelements\n%i\n', m^2); % total element number
% specify each element 
onev=ones(1,m^2);

i1=0:1:(m^2-1);
i4=m:1:(m*(m+1)-1);
for i=1:m
    for j=1:m
        ind=j+(i-1)*m;
		i1(ind) = (i-1)*(m+1) + (j-1);
		i2(ind) = (i-1)*(m+1) + j;
		i3(ind) = i2(ind) + (m+1);
		i4(ind) = i3(ind) -1;
    end
end

ielements=[i1+1; onev*3; i1; i2; i3; i4];
fprintf(fileID, '%i %i %i %i %i %i\n', ielements); % shape

% boundary 
onev2=ones(1,4*m);
% bottom     right                  top                        left
l1=[0:1:m-1	 m:m+1:(m+1)*m-1	    (m+1)^2-1:-1:m*(m+1)+1	 (m+1)*m:-m-1:m+1   ];
l2=[1:1:m    2*m+1:m+1:(m+1)^2-1    (m+1)^2-2:-1:m*(m+1)     (m+1)*(m-1):-m-1:0 ];

iboundary=[onev2; onev2; l1; l2]; 

fprintf(fileID, '\n\nboundary\n%i\n', 4*m);
fprintf(fileID, '%i %i %i %i\n', iboundary);


% vertices
fprintf(fileID, '\n\nvertices\n%i\n', (m+1)*(m+1));
fprintf(fileID, '2\n');

for j=1:m+1
	for i=1:m+1
		fprintf(fileID,'%12.8f %12.8f\n', xx(i), yy(j) );
	end
	fprintf(fileID,'\n');
end	

%for i=0:(m+1)^2-1
%    ii=mod(i,m);
%    jj=(i-ii)/m;
%    ii=ii+1; jj=jj+1;
%    fprintf(fileID,'%12.8f %12.8f\n', xx(ii), yy(jj) );
%    fprintf(fileID,'%12.8f %12.8f\n', xx(ii+1), yy(jj));
%    fprintf(fileID,'%12.8f %12.8f\n', xx(ii), yy(jj+1));
%    fprintf(fileID,'%12.8f %12.8f\n', xx(ii+1), yy(jj+1));
%    fprintf(fileID,'\n');
%end

fclose(fileID);

end
