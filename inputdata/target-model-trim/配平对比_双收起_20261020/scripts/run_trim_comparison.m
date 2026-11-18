% Run from MATLAB. This file uses the copied original model and parameter scripts.
% The aerodynamic and engine equations/tables are left unchanged.
run(fullfile(fileparts(fileparts(mfilename('fullpath'))),'source','Boeing747model.m'));
out = fileparts(fileparts(mfilename('fullpath')));
src = fullfile(out,'source');
addpath(src);
Simulink.fileGenControl('set','CacheFolder',fullfile(out,'data','cache'), ...
    'CodeGenFolder',fullfile(out,'data','codegen'),'createDir',true);
diary(fullfile(out,'data','trim_run.log'));
model = 'simulink_flightgear_control_trim_wind_5';
load_system(fullfile(src,[model '.slx']));
% Disable wind for the reference standard-day cases.
Wind.Constant.Act=0; Wind.Turblence.Act=0; Wind.Gust.Act=0; Wind.Custom.Act=0;
set_param([model '/Plane/大气环境/Constant'],'Value','0');
set_param([model '/Plane/大气环境/Manual Switch'],'sw','1');
set_param([model '/Constant4'],'Value','0');
set_param([model '/Subsystem/Constant2'],'Value','0');
set_param([model '/Subsystem/Constant4'],'Value','0');
% Expose N1 as a second independent trim input. Existing limiter is retained.
delete_line(model,'Constant5/1','Saturation2/1');
delete_block([model '/Constant5']);
add_block('simulink/Sources/In1',[model '/N1_trim'],'Port','2','Position',[1950 310 1980 340]);
add_line(model,'N1_trim/1','Saturation2/1','autorouting','on');
% Flight path angle from the actual earth-frame velocity, for level/descent constraints.
add_block('simulink/Signal Routing/From',[model '/TrimACBus'],'GotoTag','ACBus');
add_block('simulink/Signal Routing/Bus Selector',[model '/TrimVg'],'OutputSignals','Vg');
add_block('simulink/Signal Routing/Demux',[model '/TrimVgComponents'],'Outputs','3');
add_block('simulink/Math Operations/Gain',[model '/TrimUpVelocity'],'Gain','-1');
add_block('simulink/Math Operations/Divide',[model '/TrimSinGamma']);
add_block('simulink/Math Operations/Trigonometric Function',[model '/TrimAsinGamma'],'Operator','asin');
add_block('simulink/Math Operations/Gain',[model '/TrimGammaDegrees'],'Gain','180/pi');
add_block('simulink/Sinks/Out1',[model '/Gamma_deg'],'Port','3');
add_line(model,'TrimACBus/1','TrimVg/1','autorouting','on');
add_line(model,'TrimVg/1','TrimVgComponents/1','autorouting','on');
add_line(model,'TrimVgComponents/3','TrimUpVelocity/1','autorouting','on');
add_line(model,'TrimUpVelocity/1','TrimSinGamma/1','autorouting','on');
add_line(model,'From1/1','TrimSinGamma/2','autorouting','on');
add_line(model,'TrimSinGamma/1','TrimAsinGamma/1','autorouting','on');
add_line(model,'TrimAsinGamma/1','TrimGammaDegrees/1','autorouting','on');
add_line(model,'TrimGammaDegrees/1','Gamma_deg/1','autorouting','on');
% Avoid FlightGear UDP transmissions while evaluating equilibrium.
fg=find_system(model,'LookUnderMasks','all','RegExp','on','Name','^Send net_fdm Packet');
for k=1:numel(fg),set_param(fg{k},'Commented','on');end
save_system(model,fullfile(out,'B747_clean_trim_comparison.slx'));
bdclose('all');
model='B747_clean_trim_comparison';
load_system(fullfile(out,[model '.slx']));
refs=jsondecode(fileread(fullfile(out,'data','reference.json')));
comparisonRecords={}; op_saved=cell(0); reports=cell(0); templateSpec=[];
opt=findopOptions('OptimizerType','graddescent-proj','DisplayReport','off');
opt.OptimizationOptions.Algorithm='sqp';
opt.OptimizationOptions.MaxIter=400;
opt.OptimizationOptions.MaxFunEvals=10000;
opt.OptimizationOptions.TolFun=1e-10;
opt.OptimizationOptions.TolX=1e-10;
for variant=1:3
    if variant==1, case_ids=1:9; elevator_deg=2; gamma_deg=0; variant_name='original_elevator_2deg';
    elseif variant==2, case_ids=1:9; elevator_deg=0; gamma_deg=0; variant_name='neutral_elevator_0deg';
    else, case_ids=9; elevator_deg=0; gamma_deg=-1.7; variant_name='margin_note_gamma_minus1p7'; end
    for k=case_ids
        if iscell(refs), r=refs{k}; else, r=refs(k); end
        mass=r.weight_klb*1000*0.45359237;
        height=r.alt_kft*1000*0.3048;
        set_param([model '/Plane/重量//惯量/Constant5'],'Value',num2str(mass,17));
        set_param([model '/Plane/重量//惯量/Constant1'],'Value',num2str(r.cg_pct));
        set_param([model '/Constant6'],'Value',num2str(elevator_deg));
        if isempty(templateSpec), spec=operspec(model); else, spec=templateSpec; end
        theta=deg2rad(r.theta_n); alpha_body=theta-deg2rad(gamma_deg);
        tas=r.mach*sqrt(1.4*287.05287*(288.15-0.0065*height));
        spec.States(1).x=[0;theta;0];
        spec.States(1).Known=[true;false;true];
        spec.States(1).SteadyState=true(3,1);
        spec.States(2).x=[0;0;0];
        spec.States(2).Known=true(3,1);
        spec.States(2).SteadyState=true(3,1);
        spec.States(3).x=[tas*cos(alpha_body);0;tas*sin(alpha_body)];
        spec.States(3).Known=[false;true;false];
        spec.States(3).SteadyState=true(3,1);
        spec.States(4).x=[0;0;-height];
        spec.States(4).Known=true(3,1);
        spec.States(4).SteadyState=false(3,1);
        % Original trim lookup emits a value consumed as RADIANS downstream.
        physical_frl=deg2rad(r.frl_n);
        if physical_frl<0, u0=physical_frl/12; else,u0=physical_frl/4;end
        spec.Inputs(1).u=u0; spec.Inputs(1).Min=-1; spec.Inputs(1).Max=1;
        spec.Inputs(2).u=85; spec.Inputs(2).Min=60; spec.Inputs(2).Max=110;
        spec.Outputs(1).y=r.mach; spec.Outputs(1).Known=true;
        spec.Outputs(3).y=gamma_deg; spec.Outputs(3).Known=true;
        monitor_blocks={
            [model '/Plane/发动机/Subsystem/' sprintf('2-D Lookup\nTable2')],1;
            [model '/Plane/发动机/Sum'],1;
            [model '/Plane/发动机/Subsystem/Divide1'],1;
            [model '/' sprintf('1-D Lookup\nTable')],1;
            [model '/Plane/发动机/' sprintf('Bus\nSelector2')],2;
            [model '/Plane/气动系数/Subsystem58/Sum5'],1;
            [model '/Plane/重量//惯量/Constant1'],1;
            [model '/Plane/重量//惯量/Constant5'],1;
            [model '/Plane/' sprintf('Bus\nSelector8')],1;
            [model '/Plane/' sprintf('Bus\nSelector5')],1;
            [model '/Plane/气动系数/Subsystem58/Sum4'],1;
            [model '/Plane/气动系数/Subsystem58/Sum2'],1;
            [model '/Plane/气动系数/Subsystem58/Sum22'],1};
        if isempty(templateSpec)
            for j=1:size(monitor_blocks,1)
                spec=addoutputspec(spec,monitor_blocks{j,1},monitor_blocks{j,2});
                [~,warning_id]=lastwarn;
                if ~isempty(warning_id),warning('off',warning_id);end
            end
            templateSpec=spec;
        end
        fprintf('START %s %s\n',variant_name,r.id);
        [op,report]=findop(model,spec,opt);
        vals=arrayfun(@(s)s.y,report.Outputs,'UniformOutput',false);
        x=op.States(3).x;
        dx=vertcat(report.States(1:3).dx);
        frl_deg=rad2deg(vals{7});
        rr=struct('id',r.id,'variant',variant_name,'mass_kg',mass,'height_m',height, ...
            'cg_pct',r.cg_pct,'elevator_deg',elevator_deg,'gamma_target_deg',gamma_deg, ...
            'mach',vals{1},'tas_ms',vals{2},'gamma_deg',vals{3}, ...
            'epr',vals{4},'fn_total_N',vals{5},'fn_total_lbf',vals{5}/4.4482216152605, ...
            'N1_corrected_pct',vals{6},'trim_raw_rad',vals{7},'frl_deg',frl_deg, ...
            'dp_units',3-frl_deg,'cas_kt',vals{8}*1.9438444924406, ...
            'alpha_wdp_deg',rad2deg(vals{9}),'cg_actual',vals{10},'mass_actual',vals{11}, ...
            'flap_actual',vals{12},'gear_actual',vals{13}, ...
            'cl',vals{15},'cd',vals{14},'cm',vals{16}, ...
            'theta_deg',rad2deg(op.States(1).x(2)), ...
            'alpha_body_deg',rad2deg(atan2(x(3),x(1))),'u_ms',x(1),'w_ms',x(3), ...
            'trim_input',op.Inputs(1).u,'N1_pct',op.Inputs(2).u, ...
            'accel_residual_ms2',max(abs(report.States(3).dx)), ...
            'angular_accel_residual_rad_s2',max(abs(report.States(2).dx)), ...
            'steady_max_residual',max(abs(dx)),'hdot_ms',-report.States(4).dx(3), ...
            'mach_error',vals{1}-r.mach,'gamma_error',vals{3}-gamma_deg, ...
            'termination',report.TerminationString);
        rr.converged=rr.accel_residual_ms2<1e-6 && rr.angular_accel_residual_rad_s2<1e-7 && abs(rr.mach_error)<1e-7 && abs(rr.gamma_error)<1e-6;
        rr.mach_table_extrapolation=r.mach>0.9;
        rr.n1_table_extrapolation=rr.N1_corrected_pct<80 || rr.N1_corrected_pct>110;
        comparisonRecords{end+1}=rr; result=[comparisonRecords{:}];
        op_saved{end+1}=op; reports{end+1}=report;
        fprintf('RESULT %s %s dp=%.6f theta=%.6f epr=%.6f thrust=%.3f N1=%.4f residual=%.3g success=%d\n', ...
            variant_name,r.id,rr.dp_units,rr.theta_deg,rr.epr,rr.fn_total_lbf,rr.N1_pct,rr.steady_max_residual,rr.converged);
        fid=fopen(fullfile(out,'data','trim_results.json'),'w','n','UTF-8');
        fprintf(fid,'%s',jsonencode(result,'PrettyPrint',true));fclose(fid);
        save(fullfile(out,'data','trim_results.mat'),'result','op_saved','reports','refs');
    end
end
save(fullfile(out,'data','comparison_workspace.mat'));
diary off;
bdclose('all');
