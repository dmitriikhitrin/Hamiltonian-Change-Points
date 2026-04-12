function X = krn(varargin)
    % Do kron(...) on a list of inputs
    assert(nargin >= 1);
    X = varargin{1};
    for i = 2:length(varargin)
        X = kron(X, varargin{i});
    end
end