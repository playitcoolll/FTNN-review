%% 600 个 LHS 点的目标域状态配平及比剩余功率 Ps
% 直接运行本脚本。默认读取同目录 LHS_600_seed20261026.xlsx，计算全部 600 点。
% 依赖：MATLAB、Simulink、Simulink Control Design，以及目标模型所需工具箱。
% 初始化按要求运行 target-model-trim/Boeing747model.m。该文件含 clear/clc，
% 会清空 MATLAB 基础工作区；本脚本的批处理变量放在局部函数中，避免被清空。
%
% Ps 表示比剩余功率（m/s），不是大气静压：
%   Ps = dh/dt + V/g*dV/dt；本配平要求速度状态稳态，故 Ps = dh/dt = -dz/dt。
% NED 坐标 z 向下为正，爬升 Ps 为正，下降 Ps 为负。不强制水平飞行。
% 配平约束、固定 N1 及 trim 输入范围 [-1,1] 与原 traimcode.m 一致。
% 保持原模型的风场、襟翼、起落架、升降舵等设置。仅临时关闭 FlightGear 发送。
% 原始 .slx 不保存修改。失败点不删除，Ps_mps 留空，原始候选值另存。
%
% 可选：先在命令行设置配置，再运行脚本，例如只验证第 1、3 个点：
%   lhsTrimConfig = struct('RowIndices',[1 3]);
%   run('run_lhs_target_trim.m')
% 可配置字段：RowIndices（默认全部）、MaxIterations（800）、
% MaxFunctionEvaluations（20000）、SaveEvery（10）、OutputStem（默认时间戳）。
% Excel 只输出 Results 表的 7 列：Hp_m、Mach、mass_kg、deltaT_degC、
% N1_pct、Ps_mps、Success。Success=1 表示配平收敛且 trim 未触限，否则为 0。
% 同名 .mat 检查点保留详细诊断及 op/report，便于排查失败点。

if ~exist('lhsTrimConfig','var')
    lhsTrimConfig = struct();
end
lhsTrimResults = compactResults(lhsTargetTrimBatch(fileparts(mfilename('fullpath')), lhsTrimConfig));

function results = lhsTargetTrimBatch(scriptDir, cfg)
    cfg = defaults(cfg);
    rootDir = fileparts(scriptDir);
    modelDir = fullfile(rootDir, 'target-model-trim');
    model = 'simulink_flightgear_control_trim_wind_5';
    modelFile = fullfile(modelDir, [model '.slx']);
    inputFile = fullfile(scriptDir, 'LHS_600_seed20261026.xlsx');
    initFile = fullfile(modelDir, 'Boeing747model.m');
    assert(isfile(inputFile) && isfile(modelFile) && isfile(initFile), ...
        'LHS:MissingFile', '缺少 LHS 表格、目标模型或 Boeing747model.m。');
    assert(~isempty(which('findop')), 'LHS:MissingToolbox', ...
        '需要 Simulink Control Design 的 findop。');

    source = readtable(inputFile, 'Sheet','LHS数据', 'VariableNamingRule','preserve');
    expected = {'Hp, m','Mach','mass, kg','deltaT, °C','N1, %'};
    assert(all(ismember(expected, source.Properties.VariableNames)), ...
        'LHS:Columns', '输入表必须包含 Hp, m / Mach / mass, kg / deltaT, °C / N1, %%。');
    points = source{:,expected};
    lower = [1000 0.3 250000 0 70]; upper = [8000 0.8 300000 20 90];
    assert(isnumeric(points) && isequal(size(points),[600 5]) && ...
        all(isfinite(points),'all') && all(points >= lower & points <= upper,'all'), ...
        'LHS:InputRange', '输入应为 600×5 数值，且位于本次指定范围内。');
    if isempty(cfg.RowIndices), cfg.RowIndices = 1:600; end
    validateattributes(cfg.RowIndices, {'numeric'}, {'vector','integer','>=',1,'<=',600});
    assert(numel(unique(cfg.RowIndices)) == numel(cfg.RowIndices), ...
        'LHS:DuplicateRows', 'RowIndices 不应重复。');
    rowIDs = cfg.RowIndices(:); n = numel(rowIDs);
    if isempty(cfg.OutputStem)
        cfg.OutputStem = ['LHS_target_trim_Ps_' char(datetime('now','Format','yyyyMMdd_HHmmss_SSS'))];
    end
    [folderPart,~,extension] = fileparts(cfg.OutputStem);
    assert(isempty(folderPart) && isempty(extension), 'LHS:OutputStem', ...
        'OutputStem 只能是文件名主体，不能包含目录或扩展名。');
    xlsxFile = fullfile(scriptDir, [cfg.OutputStem '.xlsx']);
    matFile = fullfile(scriptDir, [cfg.OutputStem '.mat']);
    assert(~isfile(xlsxFile) && ~isfile(matFile), 'LHS:OutputExists', ...
        '输出文件已存在，请使用不同 OutputStem。');

    wasLoaded = bdIsLoaded(model);
    if wasLoaded
        assert(strcmpi(get_param(model,'FileName'),modelFile), 'LHS:WrongModel', ...
            '已加载同名的其他模型，请先关闭该模型再运行。');
        assert(strcmp(get_param(model,'Dirty'),'off'), 'LHS:UnsavedModel', ...
            '目标模型有未保存修改，请先处理这些修改再运行。');
    end
    oldDir = pwd; oldPath = path;
    environmentCleanup = onCleanup(@() restoreEnvironment(oldDir,oldPath));
    cd(modelDir); addpath(modelDir,'-begin');
    % 在基础工作区初始化 Simulink 参数；本函数的局部变量不受 clear 影响。
    fprintf('初始化目标模型：%s\n', initFile);
    evalin('base', sprintf('run(''%s'');', strrep(initFile,'''','''''')));
    load_system(modelFile);
    changed = cell(0,3);
    massBlock = [model '/Plane/重量//惯量/Constant5'];
    n1Block = [model '/Constant5'];
    temperatureBlock = [model '/Plane/大气环境/Constant'];
    remember(massBlock,'Value'); remember(n1Block,'Value'); remember(temperatureBlock,'Value');
    sendBlocks = find_system(model,'LookUnderMasks','all','RegExp','on', ...
        'Name','^Send net_fdm Packet');
    for k=1:numel(sendBlocks)
        remember(sendBlocks{k},'Commented');
    end
    % 匿名函数捕获参数值；不要在 onCleanup 中调用读取上层变量的嵌套函数。
    modelCleanup = onCleanup(@() restoreModel(model,wasLoaded,changed));
    for k=1:numel(sendBlocks)
        set_param(sendBlocks{k},'Commented','on');
    end

    options = findopOptions('OptimizerType','graddescent-proj','DisplayReport','off');
    options.OptimizationOptions.Algorithm = 'sqp';
    options.OptimizationOptions.DiffMaxChange = 0.1;
    options.OptimizationOptions.MaxIter = cfg.MaxIterations;
    options.OptimizationOptions.MaxFunEvals = cfg.MaxFunctionEvaluations;
    options.OptimizationOptions.TolFun = 1e-8;
    options.OptimizationOptions.TolX = 1e-8;
    options.OptimizationOptions.Jacobian = 'off';
    % 依据块路径定位状态，避免静默依赖状态编号。
    spec0 = operspec(model);
    statePaths = regexprep(string({spec0.States.Block}), '\s+', '');
    suffixes = ["/phithetapsi","/p,q,r","/ub,vb,wb","/xe,ye,ze"];
    idx = zeros(1,4);
    for k=1:4
        found = find(endsWith(statePaths,suffixes(k)));
        assert(isscalar(found), 'LHS:States', '无法唯一定位状态 %s。',suffixes(k));
        idx(k) = found;
    end
    assert(numel(spec0.States)==4 && numel(spec0.Inputs)==1 && numel(spec0.Outputs)==2, ...
        'LHS:ModelInterface', '当前模型接口与 traimcode.m 不一致。');
    assert(endsWith(string(spec0.Inputs(1).Block),'/trim') && ...
        endsWith(string(spec0.Outputs(1).Block),'/Output') && ...
        endsWith(string(spec0.Outputs(2).Block),'/Output1'), ...
        'LHS:Ports', '模型端口顺序与原参考代码不一致。');

    results = array2table(points(rowIDs,:), 'VariableNames', ...
        {'Hp_m','Mach','mass_kg','deltaT_degC','N1_pct'});
    results.RowID = rowIDs;
    numericNames = {'Ps_mps','Ps_candidate_mps','TAS_mps','Mach_actual', ...
        'trim_input','theta_deg','alpha_body_deg','gamma_deg', ...
        'u_mps','v_mps','w_mps','p_radps','q_radps','r_radps', ...
        'zdot_mps','speed_derivative_body_mps2','mach_error', ...
        'velocity_residual_mps2','angular_residual_radps2','attitude_residual_radps', ...
        'known_state_error','elapsed_s'};
    for k=1:numel(numericNames), results.(numericNames{k}) = nan(n,1); end
    results.Attempts = zeros(n,1);
    results.Converged = false(n,1); results.TrimAtLimit = false(n,1);
    results.Valid = false(n,1); results.Status = repmat("not_run",n,1);
    results.Termination = strings(n,1); results.ErrorMessage = strings(n,1);
    operatingPoints = cell(n,1); reports = cell(n,1);
    metadata = table( ...
        ["Input";"Model";"Initialization";"Ps definition";"Ps sign"; ...
         "Validity";"Residual tolerances";"Configuration";"Wind switch"; ...
         "Elevator constant";"Max iterations";"Max function evaluations";"Created"], ...
        [string(inputFile);string(modelFile);string(initFile); ...
         "Steady-speed specific excess power: Ps = dh/dt = -dz/dt, m/s"; ...
         "Positive = climb; negative = descent"; ...
         "Ps_mps only populated for Converged and abs(trim_input)<1-1e-6; candidate retained"; ...
         "velocity <=1e-4 m/s^2; angular/attitude <=1e-6; Mach <=1e-6; known states <=1e-6"; ...
         "Original model configuration retained; FlightGear packet sender temporarily disabled"; ...
         string(get_param([model '/Plane/大气环境/Manual Switch'],'sw')); ...
         string(get_param([model '/Constant6'],'Value')); ...
         string(cfg.MaxIterations);string(cfg.MaxFunctionEvaluations);string(datetime('now'))], ...
        'VariableNames',{'Item','Value'});
    checkpoint();
    for i=1:n
        timer = tic;
        h = results.Hp_m(i); mach = results.Mach(i);
        fprintf('[%d/%d] LHS %d: Hp=%.3f, Mach=%.5f, mass=%.3f, deltaT=%.3f, N1=%.3f\n', ...
            i,n,rowIDs(i),h,mach,results.mass_kg(i),results.deltaT_degC(i),results.N1_pct(i));
        try
            set_param(massBlock,'Value',num2str(results.mass_kg(i),17));
            set_param(n1Block,'Value',num2str(results.N1_pct(i),17));
            set_param(temperatureBlock,'Value',num2str(results.deltaT_degC(i),17));
            % 仅为初值，实际速度及 Mach 均由模型求解。范围在对流层内。
            tas0 = mach*sqrt(1.4*287.05287*(288.15-0.0065*h+results.deltaT_degC(i)));
            initialAngles = [0.05 0.12 0];
            bestScore = inf; best = []; bestOp = []; bestReport = [];
            attemptErrors = strings(0,1);
            for attempt=1:numel(initialAngles)
                results.Attempts(i) = attempt;
                try
                    spec = operspec(model);
                    a = initialAngles(attempt);
                    spec.States(idx(1)).x = [0;a;0];
                    spec.States(idx(1)).Known = [true;false;true];
                    spec.States(idx(1)).SteadyState = true(3,1);
                    spec.States(idx(2)).x = zeros(3,1);
                    spec.States(idx(2)).Known = true(3,1);
                    spec.States(idx(2)).SteadyState = true(3,1);
                    spec.States(idx(3)).x = [tas0*cos(a);0;tas0*sin(a)];
                    spec.States(idx(3)).Known = [false;true;false];
                    spec.States(idx(3)).SteadyState = true(3,1);
                    spec.States(idx(4)).x = [0;0;-h];
                    spec.States(idx(4)).Known = true(3,1);
                    spec.States(idx(4)).SteadyState = false(3,1);
                    spec.Inputs(1).u = 0; spec.Inputs(1).Known = false;
                    spec.Inputs(1).Min = -1; spec.Inputs(1).Max = 1;
                    spec.Outputs(1).y = mach; spec.Outputs(1).Known = true;
                    spec.Outputs(1).Min = -inf; spec.Outputs(1).Max = inf;
                    spec.Outputs(2).Known = false;
                    [op,report] = findop(model,spec,options);
                    candidate = evaluatePoint(op,report,idx,h,mach);
                    if isempty(best) || candidate.Score < bestScore
                        best = candidate; bestScore = candidate.Score;
                        bestOp = op; bestReport = report;
                    end
                    if candidate.Converged && ~candidate.TrimAtLimit, break; end
                catch ME
                    attemptErrors(end+1) = string(ME.identifier) + ": " + string(ME.message); %#ok<AGROW>
                end
            end
            if isempty(best)
                results.Status(i) = "error";
            else
                names = fieldnames(best);
                for k=1:numel(names)
                    if ismember(names{k},results.Properties.VariableNames)
                        results.(names{k})(i) = best.(names{k});
                    end
                end
                operatingPoints{i} = bestOp; reports{i} = bestReport;
            end
            results.ErrorMessage(i) = strjoin(attemptErrors," | ");
        catch ME
            results.Status(i) = "error";
            results.ErrorMessage(i) = string(ME.identifier) + ": " + string(ME.message);
        end
        results.elapsed_s(i) = toc(timer);
        fprintf('  %s; Ps=%g m/s; residual=%g m/s^2; elapsed=%.1f s\n', ...
            results.Status(i),results.Ps_mps(i),results.velocity_residual_mps2(i),results.elapsed_s(i));
        if mod(i,cfg.SaveEvery)==0 || i==n, checkpoint(); end
    end
    fprintf('完成：%d/%d 个点有效。结果：%s\n',sum(results.Valid),n,xlsxFile);

    function remember(block,param)
        changed(end+1,:) = {block,param,get_param(block,param)};
    end
    function checkpoint
        % 临时文件完成后再替换，避免写入中断破坏上一份检查点。
        tempMat = [tempname(scriptDir) '.mat'];
        save(tempMat,'results','operatingPoints','reports','metadata','cfg','-v7.3');
        movefile(tempMat,matFile,'f');
        tempXlsx = [tempname(scriptDir) '.xlsx'];
        writetable(compactResults(results),tempXlsx,'Sheet','Results');
        movefile(tempXlsx,xlsxFile,'f');
    end
end

function output = compactResults(results)
    output = results(:,{'Hp_m','Mach','mass_kg','deltaT_degC','N1_pct','Ps_mps','Valid'});
    output.Properties.VariableNames{7} = 'Success';
    output.Success = double(output.Success);
end

function r = evaluatePoint(op,report,idx,h,mach)
    attitude = op.States(idx(1)).x(:); rates = op.States(idx(2)).x(:);
    velocity = op.States(idx(3)).x(:); position = op.States(idx(4)).x(:);
    dv = report.States(idx(3)).dx(:); dpos = report.States(idx(4)).dx(:);
    r.TAS_mps = report.Outputs(2).y;
    r.Mach_actual = report.Outputs(1).y;
    r.trim_input = op.Inputs(1).u;
    r.theta_deg = rad2deg(attitude(2));
    r.alpha_body_deg = rad2deg(atan2(velocity(3),velocity(1)));
    r.gamma_deg = rad2deg(atan2(-dpos(3),hypot(dpos(1),dpos(2))));
    r.u_mps=velocity(1); r.v_mps=velocity(2); r.w_mps=velocity(3);
    r.p_radps=rates(1); r.q_radps=rates(2); r.r_radps=rates(3);
    r.zdot_mps = dpos(3);
    r.speed_derivative_body_mps2 = dot(velocity,dv)/max(norm(velocity),eps);
    r.Ps_candidate_mps = -dpos(3);
    r.mach_error = r.Mach_actual-mach;
    r.velocity_residual_mps2 = max(abs(dv));
    r.angular_residual_radps2 = max(abs(report.States(idx(2)).dx));
    r.attitude_residual_radps = max(abs(report.States(idx(1)).dx));
    r.known_state_error = max(abs([attitude([1 3]);rates;velocity(2);position-[0;0;-h]]));
    residuals = [r.velocity_residual_mps2/1e-4, r.angular_residual_radps2/1e-6, ...
        r.attitude_residual_radps/1e-6, abs(r.mach_error)/1e-6, r.known_state_error/1e-6];
    values = [attitude;rates;velocity;position;dv;dpos;r.TAS_mps;r.Mach_actual;r.trim_input];
    r.Converged = all(isfinite(values)) && all(isfinite(residuals)) && ...
        all(residuals<=1) && r.TAS_mps>0 && abs(r.trim_input)<=1+1e-8;
    r.TrimAtLimit = abs(r.trim_input)>=1-1e-6;
    r.Valid = r.Converged && ~r.TrimAtLimit;
    r.Ps_mps = NaN;
    if r.Valid
        r.Ps_mps = r.Ps_candidate_mps; r.Status = "ok";
    elseif r.TrimAtLimit
        r.Status = "trim_limit";
    else
        r.Status = "not_converged";
    end
    r.Termination = string(report.TerminationString);
    r.Score = max(residuals);
    if ~all(isfinite(values)) || ~isfinite(r.Score), r.Score=inf; end
    % 有效解始终优先于舵面触限解。
    if ~r.Valid, r.Score = r.Score + 1e6; end
end

function cfg = defaults(cfg)
    base = struct('RowIndices',[],'MaxIterations',800, ...
        'MaxFunctionEvaluations',20000,'SaveEvery',10,'OutputStem','');
    assert(isstruct(cfg) && isscalar(cfg),'LHS:Config','lhsTrimConfig 必须为标量结构体。');
    names = fieldnames(base);
    assert(all(ismember(fieldnames(cfg),names)), 'LHS:Config', 'lhsTrimConfig 包含未知字段。');
    for k=1:numel(names)
        if ~isfield(cfg,names{k}), cfg.(names{k})=base.(names{k}); end
    end
    validateattributes(cfg.MaxIterations,{'numeric'},{'scalar','integer','positive','finite'});
    validateattributes(cfg.MaxFunctionEvaluations,{'numeric'},{'scalar','integer','positive','finite'});
    validateattributes(cfg.SaveEvery,{'numeric'},{'scalar','integer','positive','finite'});
    cfg.OutputStem = char(cfg.OutputStem);
end

function restoreEnvironment(oldDir,oldPath)
    cd(oldDir); path(oldPath);
end

function restoreModel(model,wasLoaded,changed)
    if ~bdIsLoaded(model), return; end
    if ~wasLoaded
        close_system(model,0);
    else
        for c=size(changed,1):-1:1
            set_param(changed{c,1},changed{c,2},changed{c,3});
        end
        set_param(model,'Dirty','off');
    end
end
