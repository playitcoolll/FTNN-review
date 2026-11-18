out=fileparts(fileparts(mfilename('fullpath')));
load(fullfile(out,'data','comparison_workspace.mat'));
model='B747_clean_trim_comparison';
load_system(fullfile(out,[model '.slx']));
validation=struct([]);
indices=[1 4 9 10 18];
for n=1:numel(indices)
    idx=indices(n); rr=result(idx);
    set_param([model '/Plane/重量//惯量/Constant5'],'Value',num2str(rr.mass_kg,17));
    set_param([model '/Plane/重量//惯量/Constant1'],'Value',num2str(rr.cg_pct));
    set_param([model '/Constant6'],'Value',num2str(rr.elevator_deg));
    simIn=Simulink.SimulationInput(model);
    simIn=simIn.setInitialState(getstatestruct(op_saved{idx}));
    simIn=simIn.setExternalInput([0 rr.trim_input rr.N1_pct;2 rr.trim_input rr.N1_pct]);
    simIn=simIn.setModelParameter('StopTime','2','SaveOutput','on', ...
        'OutputSaveName','yout','SaveFormat','Dataset','Solver','ode45','MaxStep','0.02', ...
        'ReturnWorkspaceOutputs','on','SignalLogging','off');
    simOut=sim(simIn);
    ys=simOut.get('yout');
    mach_s=ys.getElement(1).Values.Data;
    tas_s=ys.getElement(2).Values.Data;
    gam_s=ys.getElement(3).Values.Data;
    check=struct('id',rr.id,'variant',rr.variant,'duration_s',2, ...
        'max_mach_drift',max(abs(mach_s-rr.mach)), ...
        'max_tas_drift_ms',max(abs(tas_s-rr.tas_ms)), ...
        'max_gamma_drift_deg',max(abs(gam_s-rr.gamma_deg)));
    check.passed=check.max_mach_drift<1e-6 && check.max_tas_drift_ms<1e-3 && check.max_gamma_drift_deg<1e-3;
    validation=[validation;check]; %#ok<AGROW>
    fprintf('VALIDATED %s %s dM=%.3g dV=%.3g dGamma=%.3g passed=%d\n',rr.id,rr.variant, ...
        check.max_mach_drift,check.max_tas_drift_ms,check.max_gamma_drift_deg,check.passed);
end
fid=fopen(fullfile(out,'data','simulation_validation.json'),'w','n','UTF-8');
fprintf(fid,'%s',jsonencode(validation,'PrettyPrint',true));fclose(fid);
save(fullfile(out,'data','simulation_validation.mat'),'validation');
bdclose('all');
assert(all([validation.passed]),'Short-time equilibrium validation did not pass');
