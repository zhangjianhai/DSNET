
load('skeletal_data.mat')
ske=cell(10,10,2);
skeletal_data_validity=zeros(10,10,2);
for i =1:10
    for j=1:10
        for k=1:2
            if numel(skeletal_data{i,j,k})~=0
                if size(skeletal_data{i,j,k}.joint_locations,3)>9
                    ske{i,j,k}=skeletal_data{i,j,k}.joint_locations;
                    skeletal_data_validity(i,j,k)=1;
                end
            end
        end
    end
    
end