function regs = combine_locals(len, minimas, w)

minimas = sort(minimas);
regs = zeros(length(minimas),2);
for i = 1:length(minimas)
    regs(i,1) = max(1,minimas(i)-w);
    regs(i,2) = min(len, minimas(i)+w);
end

l = size(regs,1);
i = 1;
while i < l 
    if regs(i,2) >= regs(i+1,1)
        regs(i,2) = regs(i+1,2);
        regs(i+1,:) = [];
        l = l - 1;
    end
    i = i + 1;
end