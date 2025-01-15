
const Bubble = ({ message }) => {

    const { content, role } = message;

    return (
        <div className={`bubble ${role}`}>
            {content}
        </div>
    )
}

export default Bubble
