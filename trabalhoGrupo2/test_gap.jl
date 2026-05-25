using AssignmentProblems
using Printf

for inst in instances(Instance)
    data = loadAssignmentProblem(inst)
    data === nothing && continue
    @printf "%-15s  agentes=%3d  jobs=%4d  lb=%8d  ub=%8d\n" \
        data.name na(data) nj(data) data.lb data.ub
end
