% Verify exported records against the original Simulink operating point reports.
out=fileparts(fileparts(mfilename('fullpath')));
load(fullfile(out,'data','trim_results.mat'),'result','op_saved','reports','refs');
assert(numel(result)==19);
for k=1:numel(result)
    % Diagnostic Sum4 is CD, Sum2 is CL (before its unchanged saturation).
    result(k).cl=reports{k}.Outputs(15).y;
    result(k).cd=reports{k}.Outputs(14).y;
    assert(abs(result(k).epr-reports{k}.Outputs(4).y)<1e-12);
    assert(abs(result(k).fn_total_N-reports{k}.Outputs(5).y)<1e-9);
    assert(abs(result(k).frl_deg-rad2deg(reports{k}.Outputs(7).y))<1e-10);
    assert(result(k).flap_actual==0 && result(k).gear_actual==0);
    assert(result(k).converged);
end
save(fullfile(out,'data','trim_results.mat'),'result','op_saved','reports','refs');
comparisonRecords=arrayfun(@(x)x,result,'UniformOutput',false);
save(fullfile(out,'data','comparison_workspace.mat'),'result','comparisonRecords','-append');
fid=fopen(fullfile(out,'data','trim_results.json'),'w','n','UTF-8');
fprintf(fid,'%s',jsonencode(result,'PrettyPrint',true));fclose(fid);
writetable(struct2table(result),fullfile(out,'data','trim_results.csv'),'Encoding','UTF-8');
fprintf('Verified 19 operating points against saved reports; all flap/gear values are zero.\n');
