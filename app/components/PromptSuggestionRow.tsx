import PromptSuggestionButton from "./PromptSuggestionButton";

const PromptSuggestionRow = ({onPromptClick}) => {

    const prompts = [
        "What is the latest news in F1?",
        "Who is the most successful F1 driver?",
        "Who is the youngest F1 driver?",
        "What is the most successful F1 team?",
        "who is the highest paid F1 driver?",
    ];

    return (
        <div className="prompt-suggestion-row">
            {prompts.map((prompt, index) => <PromptSuggestionButton
                key={`suggestion-${index}`}
                text={prompt}
                onClick={() => onPromptClick(prompt)}
            />)}
        </div>
    )
}

export default PromptSuggestionRow