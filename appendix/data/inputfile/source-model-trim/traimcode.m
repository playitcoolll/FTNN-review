%% 搜索模型 - simulink_flightgear_control_trim_wind_5 的指定工作点。
%
% 此 MATLAB 脚本等效于在稳态管理器中
% "设定" 选项卡上使用当前设定和选项。
% 它生成与点击 "配平" 按钮完全相同的工作点。

% MATLAB(R) 文件由 MATLAB(R) 9.14 和 Simulink Control Design (TM) 7.0 生成。
%
% 生成时间: 15-Jul-2026 22:19:14
bdclose('all');


%% 指定模型名称
model = 'simulink_flightgear_control_trim_wind_5';
load_system(model);

m_values = 240000:10000:300000;  % N
h_values = [457 3048 6096 9144 12192];% m 气压高度
dt_values = 70:10:90;
tem_values = [-20 -10 0 10 20];

results_ = struct('m',{},'h',{},'V',{},'mach',{},'hdot',{},'de',{},'dt',{},'tem',{},'success',{});



%% 配平计算  
for m = 1:length(tem_values)
    for l = 1:length(dt_values)
        for k = 1:length(m_values)
            for i = 1:length(h_values)
                fprintf('Processing h=%.2f\n',  h_values(i));
                set_param([model '/Plane/大气环境/Constant'], 'Value', num2str(tem_values(m)));
                set_param([model '/Constant5'], 'Value', num2str(dt_values(l)));
                set_param([model '/Plane/重量//惯量/Constant5'], 'Value', num2str(m_values(k)));

                mach_interp = [0.2 0.25 0.3 0.4 0.5 0.6 0.7 0.8 0.9];
                V_data = zeros(size(mach_interp));
                de_data = zeros(size(mach_interp));

                % aoa_interp = [0.45  0.15 0.1 0.075 0.05 0.02 0 -0.01 -0.015 -0.02 -0.025];
                % % 0.4 0.35 0.3 0.25 0.2
                % % 每组 h / W / dt / tem 重新初始化
                % V_data = zeros(size(aoa_interp));
                % de_data = zeros(size(aoa_interp));



                for j = 1:length(mach_interp)
                    if j == 1 
                    mach_init = mach_interp(j);
                    V_init =70; 
                    de_init = 0;
                    else
                    mach_init = mach_interp(j);
                    V_init = V_data(j-1); %已知
                    de_init = de_data(j-1);%初始值
                    end

                    % 创建工作点设定对象。
                    opspec = operspec(model);


                    %设置对模型中状态的约束。
                    % 状态(1) - simulink_flightgear_control_trim_wind_5/Plane/6DOF (Quaternion)/Calculate DCM & Euler Angles/phi theta psi
                    opspec.States(1).x = [0;0;0];
                    opspec.States(1).SteadyState = [true;true;true]; 
                    opspec.States(1).Known = [true;false;true];

                    % 状态(2) - simulink_flightgear_control_trim_wind_5/Plane/6DOF (Quaternion)/p,q,r
                    opspec.States(2).x = [0;0;0];
                    opspec.States(2).SteadyState = [true;true;true]; 
                    opspec.States(2).Known = [true;true;true];

                    % 状态(3) - simulink_flightgear_control_trim_wind_5/Plane/6DOF (Quaternion)/ub,vb,wb
                    opspec.States(3).x = [V_init;0;0];
                    opspec.States(3).SteadyState = [true;true;true]; 
                    opspec.States(3).Known = [false;true;false];

                    % 状态(4) - simulink_flightgear_control_trim_wind_5/Plane/6DOF (Quaternion)/xe,ye,ze
                    opspec.States(4).x = [0;0;-h_values(i)];
                    opspec.States(4).SteadyState = [false;false;false]; 
                    opspec.States(4).Known = [true;true;true]; 


                    % 输入(1) - simulink_flightgear_control_trim_wind/trim
                    opspec.Inputs(1).u = de_init;
                    opspec.Inputs(1).Min = -1;
                    opspec.Inputs(1).Max = 1;


                    % 输出(1)
                    opspec.Outputs(1).y =  mach_init;
                    opspec.Outputs(1).Known = true;
                    opspec.Outputs(1).Min = -inf;
                    opspec.Outputs(1).Max = inf;

  

                    % 创建选项
                    opt = findopOptions('OptimizerType','graddescent-proj',...
                        'DisplayReport','iter');%graddescent graddescent-proj graddescent-elim simplex lsqnonlin-proj
                    opt.OptimizationOptions.DiffMaxChange = 0.1;%每一步输入量u的变化范围最大值
                    opt.OptimizationOptions.Algorithm = 'sqp';%levenberg-marquardt  trust-region-reflective  'sqp'、'interior-point'、'trust-region-reflective' 或 'active-set'
                    opt.OptimizationOptions.MaxFunEvals = 4000000;
                    opt.OptimizationOptions.MaxIter = 4000000;
                    opt.OptimizationOptions.TolFun = 1.000000e-08;
                    opt.OptimizationOptions.TolX = 1.000000e-08;
                    opt.OptimizationOptions.Jacobian = 'off';

                    % 执行工作点搜索。
                    [op,opreport] = findop(model,opspec,opt);

                    termStr = strtrim(opreport.TerminationString);

                    if strcmp(termStr,"成功满足工作点设定。")
                        success = true;
                    else
                        success = false;
                    end

                    fprintf('Processing hdot=%.2f\n',  opreport.States(4).dx(3));

  
                    de_data(j) = op.Inputs(1).u;
                    V_data(j) = opreport.Outputs(2).y; 

                    results_(end+1) = struct(...
                        'm', m_values(k),...
                        'h', h_values(i),...
                        'V', opreport.Outputs(2).y,...     
                        'mach', opreport.Outputs(1).y,...                       
                        'hdot',opreport.States(4).dx(3), ...
                        'de',opreport.Inputs(1).u,...
                        'dt',dt_values(l),...
                        'tem',tem_values(m),...
                        'success', success ...
                        );
                end
            end
        end
    end
end

% 删除无效数据：de=±1 或 mach>0.85
invalid_idx = arrayfun(@(x) x.de == 1 || x.de == -1 , results_);
results_(invalid_idx) = [];
disp('已删除 de=±1 的无效结果。');

% T = struct2table(results_);  % 转为 table
% writetable(T, 'new_results.xlsx');





