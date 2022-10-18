function align_xlabels!(axs...)
    xspace = maximum(tight_xticklabel_spacing!, axs)
    for ax in axs
        ax.xticklabelspace = xspace
    end
end

function align_ylabels!(axs...)
    yspace = maximum(tight_yticklabel_spacing!, axs)
    for ax in axs
        ax.yticklabelspace = yspace
    end
end
